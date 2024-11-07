import itertools
import json
import re

import numpy as np
import pandas as pd

import phibase_pipeline.clean as clean
import phibase_pipeline.migrate as migrate

pd.set_option('future.no_silent_downcasting', True)


def read_phig_uniprot_mapping(path):
    return (
        pd.read_csv(path, index_col='uniprot_id')['phig_id'].to_dict()
    )


def read_phipo_chebi_mapping(path):
    return (
        pd.read_csv(path, index_col='phipo_id')
        .rename(columns={
            'chebi_id': 'id',
            'chebi_label': 'label',
        })
        .to_dict(orient='index')
    )


def uniprot_data_to_mapping(uniprot_data: pd.DataFrame) -> dict:
    uniprot_df = get_uniprot_columns(uniprot_data)
    uniprot_mapping = uniprot_df.set_index('uniprot').to_dict(orient='index')
    return uniprot_mapping


def get_genotype_data(session, genotype_id, suffix='_a'):
    genotype = session['genotypes'][genotype_id]
    if len(genotype['loci']) > 1:
        return None
    alleles = session['alleles']
    genes = session['genes']
    strain = genotype['organism_strain']
    taxon_id = genotype['organism_taxonid']
    species_name = session['organisms'][str(taxon_id)]['full_name']
    uniprot_id = None
    allele_display_name = None
    for locus in genotype['loci']:
        for locus_allele in locus:
            expression = locus_allele.get('expression')
            expr_str = f' [{expression}]' if expression else ''
            allele_id = locus_allele['id']
            allele = alleles[allele_id]
            description = allele.get('description')
            desc_str = f'({description})' if description else ''
            allele_type = allele['allele_type'].replace('_', ' ')
            allele_display_name = (
                f"{allele['name']}{desc_str} ({allele_type}){expr_str}"
            )
            gene_id = allele['gene']
            gene = genes[gene_id]
            uniprot_id = gene['uniquename']
    record = {
        'uniprot': uniprot_id,
        'taxid': taxon_id,
        'organism': species_name,
        'strain': strain,
        'modification': allele_display_name,
    }
    suffixed = {k + suffix: v for k, v in record.items()}
    return suffixed


def get_physical_interaction_data(session, gene_id, interacting_gene_id):
    genes = session['genes']
    organisms = session['organisms']
    data = {}
    for gid, suffix in ((gene_id, '_a'), (interacting_gene_id, '_b')):
        gene = genes[gid]
        uniprot_id = gene['uniquename']
        organism_name = gene['organism']
        taxid = next(
            taxid for taxid, organism in organisms.items()
            if organism['full_name'] == organism_name
        )
        gene_data = {
            'uniprot': uniprot_id,
            'taxid': int(taxid),
            'organism': organism_name,
            'strain': None,
            'modification': None,
        }
        data.update({k + suffix: v for k, v in gene_data.items()})
    return data


def get_metagenotype_data(session, metagenotype_id):
    metagenotype = session['metagenotypes'][metagenotype_id]
    pathogen_genotype_id = metagenotype['pathogen_genotype']
    host_genotype_id = metagenotype['host_genotype']
    pathogen_columns = get_genotype_data(
        session, pathogen_genotype_id, suffix='_a'
    )
    if pathogen_columns is None:
        return None
    host_columns = get_genotype_data(
        session, host_genotype_id, suffix='_b'
    )
    if host_columns is None:
        return None
    return {**pathogen_columns, **host_columns}


def get_tissue_id_str(annotation):
    tissue_ids = [
        ext['rangeValue']
        for ext in annotation.get('extension', [])
        if ext['relation'] == 'infects_tissue'
    ]
    tissue_id_str = '; '.join(tissue_ids)
    return tissue_id_str or None


def get_canto_columns(canto_export: dict, effector_ids: set[str]) -> pd.DataFrame:
    column_order = [
        'uniprot_a',
        'taxid_a',
        'organism_a',
        'strain_a',
        'modification_a',
        'uniprot_b',
        'taxid_b',
        'organism_b',
        'strain_b',
        'modification_b',
        'phenotype',
        'disease',
        'host_tissue',
        'evidence_code',
        'interaction_type',
        'pmid',
        'high_level_terms',
    ]
    interaction_type_map = {
        'disease_name': 'interspecies interaction',
        'gene_for_gene_phenotype': 'gene-for-gene interaction',
        'pathogen_host_interaction_phenotype': 'interspecies interaction',
        'physical_interaction': 'protein-protein interaction',
    }
    curation_sessions = canto_export['curation_sessions']
    records = []
    for session in curation_sessions.values():
        pmid = int(next(iter(session['publications'].keys())).replace('PMID:', ''))
        for annotation in session.get('annotations', []):
            data = None
            if metagenotype_id := annotation.get('metagenotype'):
                data = get_metagenotype_data(session, metagenotype_id)
                if data is None:
                    continue  # Multi-allele genotypes not handled yet
                is_disease = annotation['type'] == 'disease_name'
                term_id = annotation['term']
                phenotype = None if is_disease else term_id
                disease = term_id if is_disease else None
                tissue_ids = get_tissue_id_str(annotation)
                high_level_terms = get_high_level_terms(annotation)
                has_effector_gene = (
                    data['uniprot_a'] in effector_ids
                    or data['uniprot_b'] in effector_ids
                )
                if has_effector_gene:
                    high_level_terms.extend(['Effector'])
                high_level_term_str = (
                    '; '.join(high_level_terms) if high_level_terms else None
                )
            elif annotation['type'] == 'physical_interaction':
                data = get_physical_interaction_data(
                    session,
                    gene_id=annotation['gene'],
                    interacting_gene_id=annotation['interacting_genes'][0],
                )
                phenotype = None
                disease = None
                tissue_ids = None
                high_level_term_str = None

            if data is None:
                continue
            interaction_type = interaction_type_map.get(annotation['type'])
            # Sometimes the evidence_code is an empty string instead of null
            evidence_code = annotation.get('evidence_code')
            if evidence_code == '':
                evidence_code = None
            records.append(
                {
                    **data,
                    'phenotype': phenotype,
                    'disease': disease,
                    'host_tissue': tissue_ids,
                    'evidence_code': evidence_code,
                    'interaction_type': interaction_type,
                    'pmid': pmid,
                    'high_level_terms': high_level_term_str,
                }
            )
    return pd.DataFrame.from_records(records).fillna(np.nan)[column_order]


def get_uniprot_columns(uniprot_data):

    def get_ensembl_id_column(df):
        ensembl_columns = df.columns[df.columns.str.startswith('ensembl_')]
        # DataFrame.sum is only safe due to no overlap between columns
        combined_column = (
            df[ensembl_columns]
            .fillna('')
            .sum(axis=1)
            .replace('', np.nan)
            .str.rstrip(';')  # UniProt uses trailing semicolon
            .str.replace(';', '; ')
        )
        return combined_column

    renames = {
        'Entry': 'uniprot',
        'Organism (ID)': 'taxid',
        'Taxonomic lineage (Ids)': 'lineage',
        'Ensembl': 'ensembl_genomes',
        'EnsemblBacteria': 'ensembl_bacteria',
        'EnsemblFungi': 'ensembl_fungi',
        'EnsemblMetazoa': 'ensembl_metazoa',
        'EnsemblPlants': 'ensembl_plants',
        'EnsemblProtists': 'ensembl_protists',
    }
    columns = list(renames.keys())
    df = uniprot_data[columns].rename(columns=renames).copy()
    df['uniprot_matches'] = 'none'
    species_taxids = (
        df.lineage
        .str.extract(r'(\d+) \(species\)', expand=False)
        .astype('Int64')
    )
    is_strain_taxid = species_taxids.notna()
    df['taxid_species'] = df.taxid.mask(is_strain_taxid, species_taxids)
    df['taxid_strain'] = df.taxid.where(is_strain_taxid).astype('Int64')
    uniprot_match_values = {
        'strain': df.taxid_strain.notna(),
        'species': df.taxid_strain.isna(),
    }
    for value, cond in uniprot_match_values.items():
        df.loc[cond, 'uniprot_matches'] = value
    df['ensembl'] = get_ensembl_id_column(df)
    columns = [
        'uniprot', 'ensembl', 'taxid_species', 'taxid_strain', 'uniprot_matches'
    ]
    return df[columns]


def add_uniprot_columns(canto_df, uniprot_df):
    column_order = [
        'phibase_id',
        'uniprot_a',
        'ensembl_a',
        'taxid_species_a',
        'organism_a',
        'taxid_strain_a',
        'strain_a',
        'uniprot_matches_a',
        'modification_a',
        'uniprot_b',
        'ensembl_b',
        'taxid_species_b',
        'organism_b',
        'taxid_strain_b',
        'strain_b',
        'uniprot_matches_b',
        'modification_b',
        'phenotype',
        'disease',
        'host_tissue',
        'evidence_code',
        'interaction_type',
        'pmid',
        'high_level_terms',
    ]
    uniprot_a_df, uniprot_b_df = (
        (
            uniprot_df
            .rename(columns=lambda s: s + f'{suffix}')
            .set_index(f'uniprot{suffix}')
        )
        for suffix in ('_a', '_b')
    )
    merge_args = {'right_index': True, 'how': 'left'}
    merged_df = (
        canto_df
        .merge(uniprot_a_df, left_on='uniprot_a', **merge_args)
        .merge(uniprot_b_df, left_on='uniprot_b', **merge_args)
    )
    merged_df.taxid_species_a = merged_df.taxid_species_a.mask(
        merged_df.uniprot_a.isna(), merged_df.taxid_a
    )
    merged_df.taxid_species_b = merged_df.taxid_species_b.mask(
        merged_df.uniprot_b.isna(), merged_df.taxid_b
    )
    # Canto doesn't export PHI-base 4 IDs but we need the column for ordering.
    if 'phibase_id' not in canto_df.columns:
        merged_df['phibase_id'] = pd.Series(dtype=object)
    merged_df.taxid_strain_a = merged_df.taxid_strain_a.astype('Int64')
    merged_df.taxid_strain_b = merged_df.taxid_strain_b.astype('Int64')
    # TODO: Fix this in the get_uniprot_columns
    merged_df.uniprot_matches_a = merged_df.uniprot_matches_a.fillna('none')
    merged_df.uniprot_matches_b = merged_df.uniprot_matches_b.fillna('none')
    # These columns aren't needed since Ensembl are loading these IDs by
    # using the UniProtKB accession number.
    # TODO: Don't load these columns from UniProt
    merged_df.ensembl_a = np.nan
    merged_df.ensembl_b = np.nan
    return merged_df[column_order]


def get_high_level_terms(annotation):
    term_mapping = {
        'PHIPO:0000022': 'Resistance to chemical',
        'PHIPO:0000021': 'Sensitivity to chemical',
        'PHIPO:0000513': 'Lethal',
    }
    extension_mapping = {
        'PHIPO:0000207': 'Loss of mutualism',
        'PHIPO:0000014': 'Increased virulence',
        'PHIPO:0000010': 'Loss of pathogenicity',
        'PHIPO:0000015': 'Reduced virulence',
        'PHIPO:0000004': 'Unaffected pathogenicity',
    }
    high_level_terms = set()
    primary_term_id = annotation.get('term')
    if not primary_term_id:
        return []
    hlt_1 = term_mapping.get(primary_term_id)
    if hlt_1:
        high_level_terms.add(hlt_1)
    for extension in annotation.get('extension', []):
        ext_term_id = extension['rangeValue']
        hlt_2 = extension_mapping.get(ext_term_id)
        if hlt_2:
            high_level_terms.add(hlt_2)
    return list(sorted(high_level_terms))


def get_effector_gene_ids(canto_export):
    EFFECTOR_TERM_ID = 'GO:0140418'
    uniprot_ids = set()
    for session in canto_export['curation_sessions'].values():
        genes = session.get('genes')
        if not genes:
            continue
        for annotation in session.get('annotations', []):
            if annotation.get('term') == EFFECTOR_TERM_ID:
                gene_id = annotation['gene']
                uniprot_id = genes[gene_id]['uniquename']
                uniprot_ids.add(uniprot_id)
    return uniprot_ids


def make_ensembl_canto_export(
    canto_export: dict, uniprot_data: pd.DataFrame, phig_mapping: dict
) -> pd.DataFrame:
    uniprot_df = get_uniprot_columns(uniprot_data)
    effector_ids = get_effector_gene_ids(canto_export)
    canto_df = get_canto_columns(canto_export, effector_ids)
    combined_df = add_uniprot_columns(canto_df, uniprot_df)
    # Add empty columns for compatibility with Ensembl's pipeline
    combined_df[['sequence_A', 'sequence_B', 'buffer_col']] = np.nan
    combined_df = combined_df.rename(columns={'phibase_id': 'phig_id'})
    # TODO: Move this to get_canto_columns
    combined_df.phig_id = [
        phig_mapping.get(x, np.nan) for x in combined_df.uniprot_a.values
    ]
    return combined_df


def get_amr_records(canto_export, phig_mapping, chebi_mapping):
    curation_sessions = canto_export['curation_sessions']
    effector_ids = get_effector_gene_ids(canto_export)
    records = []
    for session in curation_sessions.values():
        for annotation in session.get('annotations', []):
            chebi_term = chebi_mapping.get(annotation.get('term'))
            if not chebi_term:
                continue  # Can't include data without a chemical ID

            genotype_id = annotation.get('genotype')
            if not genotype_id:
                continue  # Assuming chemical annotations are only on genotypes

            genotype = session['genotypes'][genotype_id]
            if len(genotype['loci']) > 1:
                continue  # Multi-allele genotypes are not yet supported by Ensembl

            genotype_data = get_genotype_data(session, genotype_id)
            uniprot_id = genotype_data['uniprot_a']
            phig_id = phig_mapping.get(uniprot_id)
            if not phig_id:
                continue  # Can't include data without a cross-reference to PHI-base

            high_level_terms = get_high_level_terms(annotation)
            if uniprot_id in effector_ids:
                high_level_terms.append('Effector')
            high_level_terms_str = (
                '; '.join(high_level_terms) if high_level_terms else None
            )

            annotation_type = annotation['type'].replace('pathogen_phenotype', 'antimicrobial_interaction')
            pmid_number = lambda pmid: int(pmid.replace('PMID:', ''))
            row = {
                'phig_id': phig_id,
                'interactor_A_molecular_id': uniprot_id,
                'ensembl_a': None,  # leave empty
                'taxid_species_a': genotype_data['taxid_a'],
                'organism_a': genotype_data['organism_a'],
                'taxid_strain_a': None,  # leave empty
                'strain_a': genotype_data['strain_a'],
                'uniprot_matches_a': None,  # leave empty
                'modification_a': genotype_data['modification_a'],
                'interactor_B_molecular_id': chebi_term['id'],
                'ensembl_b': None,  # leave empty
                'taxid_species_b': None,  # leave empty
                'organism_b': chebi_term['label'],
                'taxid_strain_b': None,  # leave empty
                'strain_b': None,  # leave empty
                'uniprot_matches_b': None,  # leave empty
                'modification_b': None,  # can't modify ChEBI molecules
                'phenotype': annotation['term'],
                'disease': None,  # disease not applicable
                'host_tissue': get_tissue_id_str(annotation),
                'evidence_code': annotation['evidence_code'],
                'interaction_type': annotation_type,
                'pmid': pmid_number(annotation['publication']),
                'high_level_terms': high_level_terms_str,
                'interactor_A_sequence': None,
                'interactor_B_sequence': None,
                'buffer_col': None,
            }
            records.append(row)
    return records


def merge_amr_uniprot_data(amr_records, uniprot_mapping):
    merged = []
    for record in amr_records:
        uniprot_id = record['interactor_A_molecular_id']
        uniprot_data = uniprot_mapping.get(uniprot_id)
        if uniprot_data is None:
            continue  # Maybe this should be an error
        uniprot_data_suffixed = {k + '_a': v for k, v in uniprot_data.items()}
        merged_record = record.copy()
        merged_record.update(uniprot_data_suffixed)
        merged.append(merged_record)
    return merged


def make_ensembl_amr_export(
    canto_export: dict,
    *,
    chebi_mapping: dict,
    phig_mapping: dict,
    uniprot_mapping: dict,
) -> pd.DataFrame:
    amr_records = get_amr_records(canto_export, phig_mapping, chebi_mapping)
    merged_records = merge_amr_uniprot_data(amr_records, uniprot_mapping)
    phig_numbers = lambda series: series.str.slice(len('PHIG:')).astype(int)
    amr_df = (
        pd.DataFrame.from_records(merged_records)
        .drop_duplicates()
        .sort_values('phig_id', key=phig_numbers)
    )
    # Use Int64 to avoid rendering numbers as floating point
    amr_df['taxid_strain_a'] = amr_df['taxid_strain_a'].astype('Int64')
    return amr_df.fillna(np.nan)


def make_ensembl_phibase_export(
    phi_df: pd.DataFrame,
    *,
    uniprot_data: pd.DataFrame,
    phenotype_mapping,
    disease_mapping,
    tissue_mapping,
    in_vitro_growth_classifier,
    exp_tech_mapping: pd.DataFrame,
) -> pd.DataFrame:

    def get_phenotype_column(phi_df, phenotype_mapping):
        """
        Map all phenotype strings to ontology term IDs in each phenotype
        column in phi_df, then concatenate the IDs into one string per row.
        """
        phenotype_columns = list(phenotype_mapping.column_1.unique())
        df = phi_df[phenotype_columns].dropna(axis=1).copy()
        # Add this column back in since it's needed for mapping
        df['is_filamentous'] = phi_df.is_filamentous
        selector = pd.MultiIndex.from_tuples(
            tuple(df.transpose()[0].to_dict().items())
        )
        phenotype_mapping = phenotype_mapping.set_index(['column_1', 'value_1'])
        # Use intersection to drop indexes that will cause a KeyError
        selector = phenotype_mapping.index.intersection(selector)
        phenotype_mapping = (
            phenotype_mapping.loc[selector].reset_index(drop=False)
        )
        columns = [
            'column_1',
            'value_1',
            'column_2',
            'value_2',
            'primary_id',
            'extension_range',
        ]
        column_map = (
            phenotype_mapping[columns]
            .replace(np.nan, None)
            .to_dict(orient='records')
        )

        single_map = [d for d in column_map if d['column_2'] is None]
        single_value_map = {
            k: {
                v['value_1']:  v['primary_id'] if k != 'mutant_phenotype' else v['extension_range']
                for v in g
            }
            for k, g in itertools.groupby(single_map, key=lambda d: d['column_1'])
        }
        multi_map = [d for d in column_map if d['column_2'] is not None]

        # We don't need the extension_range handling here because no
        # multi-criteria mapping uses that field (yet).
        grouping_order = ('column_1', 'column_2', 'value_2')
        sorter = lambda d: [d[k] for k in grouping_order]
        g0 = sorted(multi_map, key=sorter)
        multi_value_map = {
            k1: {k2: {k3: {d['value_1']: d['primary_id'] for d in g3}}}
            for k1, g1 in itertools.groupby(g0, lambda d: d['column_1'])
            for k2, g2 in itertools.groupby(g1, lambda d: d['column_2'])
            for k3, g3 in itertools.groupby(g2, lambda d: d['value_2'])
        }

        for column, mapping in single_value_map.items():
            df[column] = df[column].map(mapping)

        for column_1, level_1 in multi_value_map.items():
            for column_2, replace_2 in level_1.items():
                for predicate, replacements in replace_2.items():
                    index = df[column_2] == predicate
                    replaced = df.loc[index, column_1].map(replacements)
                    df.loc[index, column_1] = replaced

        phenotypes = (
            df[phenotype_columns[0]]
            .astype(object)
            .str.cat(df[phenotype_columns[1:]], na_rep='', sep=';')
            .str.strip(';')
            .str.replace(';+', '; ', regex=True)
        )
        return phenotypes

    def map_experimental_techniques(phi_df, exp_tech_df):
        """Split stable experimental techniques into allele types and evidence columns."""
        exp_tech_df = exp_tech_df.set_index('exp_technique_stable')[
            ['pathogen_allele_type', 'host_allele_type', 'evidence_code']
        ]
        phi_df.exp_technique_stable = (
            phi_df.exp_technique_stable.str.lower().str.replace('\xa0', ' ')
        )
        phi_df = phi_df.merge(
            exp_tech_df,
            left_on='exp_technique_stable',
            right_index=True,
            how='left',
        )
        # TODO: Add transient expression experimental evidence?
        return phi_df

    phi_df = phi_df.copy()
    phi_df.columns = clean.get_normalized_column_names(phi_df)

    phi_df['host_uniprot'] = phi_df.host_target_id.str.extract(
        r'uniprot: (.+?);', flags=re.I, expand=False
    )
    phi_df = map_experimental_techniques(phi_df, exp_tech_mapping)

    column_renames = {
        'phi_molconn_id': 'phibase_id',
        'protein_id': 'uniprot_a',
        'pathogen_id': 'taxid_a',
        'pathogen_species': 'organism_a',
        'pathogen_strain': 'strain_a',
        'pathogen_allele_type': 'modification_a',
        'host_uniprot': 'uniprot_b',
        'host_id': 'taxid_b',
        'host_species': 'organism_b',
        'host_strain': 'strain_b',
        'host_allele_type': 'modification_b',
        'disease': 'disease',
        'tissue': 'host_tissue',
        'evidence_code': 'evidence_code',
        'pmid': 'pmid',
        'mutant_phenotype': 'high_level_terms',
    }
    columns = list(column_renames)
    export_df = phi_df[columns].rename(columns=column_renames)

    # Don't need to classify other interaction types from PHI-base 4 yet
    export_df['interaction_type'] = 'interspecies interaction'

    # Guard against an empty merge column causing an error
    export_df['phenotype'] = pd.Series(dtype='object')

    uniprot_df = get_uniprot_columns(uniprot_data)
    export_df = add_uniprot_columns(export_df, uniprot_df)
    phi_df = migrate.add_filamentous_classifier_column(
        in_vitro_growth_classifier, phi_df
    )
    export_df['phenotype'] = get_phenotype_column(phi_df, phenotype_mapping)

    # TODO: Change add_disease_term_ids to return a series
    export_df = migrate.add_disease_term_ids(disease_mapping, export_df)
    export_df.disease = export_df.disease_id
    export_df = export_df.drop('disease_id', axis=1)

    export_df.host_tissue = migrate.get_tissue_id_series(
        tissue_mapping, export_df.rename(columns={'host_tissue': 'tissue'})
    )
    # TODO: Unify this with existing code in clean.py
    common_name_pattern = r'\s*\(.+$'
    export_df.organism_b = export_df.organism_b.str.replace(
        common_name_pattern, '', regex=True
    )
    export_df.high_level_terms = export_df.high_level_terms.str.capitalize()
    return export_df


def make_ensembl_exports(
    phi_df: pd.DataFrame,
    canto_export: dict,
    *,
    uniprot_data: pd.DataFrame,
    phig_mapping: dict,
    chebi_mapping: dict,
    phenotype_mapping,
    disease_mapping,
    tissue_mapping,
    in_vitro_growth_classifier,
    exp_tech_mapping: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    phibase4_export = make_ensembl_phibase_export(
        phi_df,
        uniprot_data=uniprot_data,
        phenotype_mapping=phenotype_mapping,
        disease_mapping=disease_mapping,
        tissue_mapping=tissue_mapping,
        in_vitro_growth_classifier=in_vitro_growth_classifier,
        exp_tech_mapping=exp_tech_mapping,
    )
    phibase5_export = make_ensembl_canto_export(
        canto_export,
        uniprot_data,
        phig_mapping,
    )
    amr_export = make_ensembl_amr_export(
        canto_export,
        phig_mapping=phig_mapping,
        uniprot_mapping=uniprot_data_to_mapping(uniprot_data),
        chebi_mapping=chebi_mapping,
    )
    return {
        'phibase4': phibase4_export,
        'phibase5': phibase5_export,
        'amr': amr_export,
    }
