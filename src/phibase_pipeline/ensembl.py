import json

import numpy as np
import pandas as pd


pd.set_option('future.no_silent_downcasting', True)


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
            # Note the deliberate leading space!
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


def get_canto_columns(canto_export: dict, effector_ids: set[str]) -> pd.DataFrame:

    def get_tissue_ids(annotation):
        tissue_ids = [
            ext['rangeValue']
            for ext in annotation.get('extension', [])
            if ext['relation'] == 'infects_tissue'
        ]
        tissue_id_str = '; '.join(tissue_ids)
        return tissue_id_str or None

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
                tissue_ids = get_tissue_ids(annotation)
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


def combine_canto_uniprot_data(canto_df, uniprot_df):
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
    merged_df['phibase_id'] = pd.Series(dtype='object')
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


def make_ensembl_canto_export(export_path, uniprot_data_path, out_path):
    with open(export_path, encoding='utf-8') as export_file:
        canto_export = json.load(export_file)
    uniprot_data_df = pd.read_csv(uniprot_data_path)
    uniprot_df = get_uniprot_columns(uniprot_data_df)
    effector_ids = get_effector_gene_ids(canto_export)
    canto_df = get_canto_columns(canto_export, effector_ids)
    combined_df = combine_canto_uniprot_data(canto_df, uniprot_df)
    combined_df.to_csv(out_path, index=False)
