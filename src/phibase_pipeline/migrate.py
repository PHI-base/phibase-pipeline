#!/usr/bin/env python3
# coding: utf-8

import hashlib
import itertools
import json
import os
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from phibase_pipeline.clean import clean_phibase_csv
from phibase_pipeline.wild_type import get_all_feature_mappings, get_wt_features


DATA_DIR = Path(__file__).parent / 'data'


def load_bto_id_mapping(path):
    renames = {
        'LABEL': 'term_label',
        'ID': 'term_id',
    }
    return pd.read_csv(path).rename(columns=renames).set_index('term_label')['term_id']


def load_phipo_mapping(path):
    return pd.read_csv(path).set_index('ID')['LABEL'].to_dict()


def load_json(path):
    with open(path, encoding='utf8') as json_file:
        return json.load(json_file)


def load_exp_tech_mapping(path):
    def join_rows(column):
        if column.size == 1:
            return column
        if column.isna().all():
            return np.nan
        else:
            return ' | '.join(column.dropna().unique())

    unique_columns = ['exp_technique_stable', 'gene_protein_modification']
    df = pd.read_csv(path).groupby(unique_columns, dropna=False).agg(join_rows)
    valid_rows = (df != '?').all(axis=1)
    assert ~df.index.duplicated().any()
    return df[valid_rows].reset_index()


def load_phenotype_column_mapping(path):
    df = pd.read_csv(path, na_values=['?'], dtype='str')
    for column in df.columns:
        df[column] = df[column].str.strip()
    has_primary_id = df.primary_id.notna()
    has_extension = df.extension_range.notna() & df.extension_relation.notna()
    return df[has_primary_id | has_extension]


def load_disease_column_mapping(phido_path, extra_path):
    term_id_pattern = re.compile(r'^(?:obo:)?PHIDO_(\d{7})$')
    renames = {
        'ID': 'term_id',
        'LABEL': 'term_label',
    }
    phido_df = pd.read_csv(phido_path).rename(columns=renames)
    phido_df = phido_df[phido_df.term_id.str.match(term_id_pattern)]
    obo_term_ids = phido_df.term_id.str.replace(term_id_pattern, r'PHIDO:\1')
    phido_df.term_id = obo_term_ids
    phido_df.term_label = phido_df.term_label.str.lower()
    extra_df = pd.read_csv(extra_path)
    combined_df = pd.concat([phido_df[['term_label', 'term_id']], extra_df])
    mapping = combined_df.set_index('term_label')['term_id'].to_dict()
    return mapping


def load_in_vitro_growth_classifier(path):
    df = pd.read_csv(path)
    df.is_filamentous = df.is_filamentous.str.lower()
    classifier_column = (
        df.set_index('ncbi_taxid').is_filamentous
        .loc[lambda x: x.isin(('yes', 'no'))]
        .replace({'yes': True, 'no': False})
    )
    return classifier_column


def split_compound_rows(df, column, sep='; '):
    original_columns = df.columns
    split_df = df[column].str.split(sep, regex=False).explode().to_frame()
    rest_df = df.drop(column, axis=1)
    return split_df.join(rest_df)[original_columns].reset_index(drop=True)


def split_compound_columns(df):
    separator_column_map = {
        '; ': ('pathogen_strain', 'host_strain'),
        ' | ': ('name', 'allele_type', 'description'),
    }
    for sep, columns in separator_column_map.items():
        for column in columns:
            df = split_compound_rows(df, column, sep)
    return df


def get_approved_pmids(canto_export):
    prefix_len = len('PMID:')
    pmid_number = lambda pmid: int(pmid[prefix_len:])
    get_pmid = lambda session: next(iter(session['publications']))
    return set(
        pmid_number(get_pmid(session))
        for session in canto_export['curation_sessions'].values()
        if session['metadata']['annotation_status'] == 'APPROVED'
    )


def filter_phi_df(phi_df, approved_pmids=None):
    def filter_columns(phi_df):
        columns_exclude = {
            'chr_location',
            'host_description',
            'database',
            'pathway',
            'species_expert',
            'ref_source',
            'doi',
            'ref_detail',
            'author_email',
            'author_reference',
            'year',
            'curation_details',
            'file_name',
            'batch_no',
            'phi3_pars',
        }
        columns_include = [c for c in phi_df.columns if c not in columns_exclude]
        return phi_df[columns_include]

    def filter_rows(phi_df, approved_pmids):
        # Regular expression for a valid UniProtKB accession number.
        uniprot_pattern = re.compile(
            r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$'
        )
        # Every row must have a PMID since PHI-Canto requires it.
        rows_with_pmid = phi_df['pmid'].notna()
        # PHI-Canto doesn't support any protein IDs besides UniProt.
        rows_in_uniprot = phi_df['protein_id_source'].notna()
        # Only include UniProt accession numbers that are valid.
        rows_with_valid_uniprot_id = phi_df['protein_id'].str.match(
            uniprot_pattern, na=False
        )
        # Only include rows where the host is at the species level (or below).
        # Note that 'Lethal pathogen phenotype' rows will not be filtered out.
        rows_without_host_genus = ~phi_df.host_species.str.match(r'^[A-Z][a-z]+$')
        rows_include = (
            rows_with_pmid
            & rows_in_uniprot
            & rows_with_valid_uniprot_id
            & rows_without_host_genus
        )
        if approved_pmids:
            # PHI-Canto curation supersedes curation in PHI-base 4, so exclude any
            # rows with PMIDs that exist in the export of approved curation sessions.
            rows_not_already_curated = ~phi_df.pmid.isin(approved_pmids)
            rows_include = rows_include & rows_not_already_curated
        return phi_df.loc[rows_include]

    return filter_columns(filter_rows(phi_df, approved_pmids))


def add_alleles_and_evidence(exp_tech_df, phi_df):
    merge_columns = ['exp_technique_stable', 'gene_protein_modification']
    columns = ['allele_type', 'name', 'description', 'evidence_code', 'expression']
    return phi_df.merge(
        exp_tech_df.set_index(merge_columns)[columns],
        left_on=merge_columns,
        right_index=True,
        how='left',
    )


def add_session_ids(df):
    """Generate fake curation session IDs for publications in PHI-base by
    truncating a SHA-256 hash of the PMID number to 16 characters."""
    df['session_id'] = (
        df.pmid
        .astype(str)
        .str.encode('ascii')
        .apply(lambda x: hashlib.sha256(x).hexdigest())
        .str.slice(stop=16)
    )
    return df


def add_gene_ids(df):
    def add_pathogen_gene_ids(df):
        df['canto_pathogen_gene_id'] = df.pathogen_species + ' ' + df.protein_id
        return df

    def add_host_gene_ids(df):
        host_gene_ids = df.host_genotype_id.str.extract(
            'UniProt: ([A-Z0-9]+)', expand=False
        )
        df['canto_host_gene_id'] = df.host_species + ' ' + host_gene_ids
        return df

    return add_pathogen_gene_ids(add_host_gene_ids(df))


def add_allele_ids(df):
    columns = ['session_id', 'protein_id', 'allele_type', 'name', 'description']
    df = df.sort_values(columns)
    allele_ids = []
    i = 1

    for current_row, next_row in itertools.pairwise(df[columns].values):
        session_id, protein_id, *props = current_row
        next_session_id, next_protein_id, *next_props = next_row
        allele_ids.append(f'{protein_id}:{session_id}-{i}')
        if (next_session_id != session_id) | (protein_id != next_protein_id):
            i = 1
        elif props != next_props:
            i += 1

    # Append the final row because iteration stops early.
    allele_ids.append(f'{next_protein_id}:{next_session_id}-{i}')
    df['pathogen_allele_id'] = allele_ids
    df = df.sort_index()  # restore original sort order
    return df


def add_genotype_ids(df):
    def add_ids(df, columns, indexer, append_ids=False):
        values = (
            df.loc[indexer, columns]
            .reset_index(drop=False)
            .sort_values(columns)
            .values
        )
        i_offset = 0
        if append_ids:
            # The genotype IDs for multi-locus genotypes need to start after the
            # end of the ID range for single locus genotypes.
            session_genotype_counts = (
                df.groupby('session_id').pathogen_genotype_n.agg(max).to_dict()
            )
            # Initialize the genotype ID offset based on the first row
            i_offset = session_genotype_counts[df.iloc[0].session_id]

        indexes = []
        genotype_nums = []
        i = 1 + i_offset

        for current_row, next_row in itertools.pairwise(values):
            index, session_id, *props = current_row
            next_index, next_session_id, *next_props = next_row
            indexes.append(index)
            genotype_nums.append(i)
            if next_session_id != session_id:
                if append_ids:
                    i_offset = session_genotype_counts[next_session_id]
                i = 1 + i_offset
            elif props != next_props:
                i += 1

        # Append the final row because pairwise iteration stops early
        indexes.append(next_index)
        genotype_nums.append(i)

        df.loc[indexes, 'pathogen_genotype_n'] = genotype_nums
        df.pathogen_genotype_n = df.pathogen_genotype_n.fillna(0).astype(int)
        df.loc[indexes, 'pathogen_genotype_id'] = (
            df.session_id + '-genotype-' + df.pathogen_genotype_n.astype(str)
        )
        return df

    def fill_multiple_mutation_ids(df):
        # Ensure all multiple mutations are symmetric and reflexive.
        indexer = df.multiple_mutation.notna()
        df2 = df.loc[indexer]
        sep = '; '
        multi_mut_ids = (
            df2.phi_molconn_id.str.cat(df2.multiple_mutation, sep)
            .str.split(sep)
            .apply(lambda x: sorted(set(x)))
            .str.join(sep)
        )
        df.loc[indexer, 'multiple_mutation'] = multi_mut_ids
        return df

    def add_wt_host_genotype_ids(df):
        # Do not create host genotype IDs for rows with no host taxon ID
        has_host_id = df.host_id.notna()
        species = df.loc[has_host_id, 'host_species']
        strains = df.loc[has_host_id, 'host_strain'].fillna('Unknown strain')
        df.loc[has_host_id, 'canto_host_genotype_id'] = (
            species + '-wild-type-genotype-' + strains
        ).str.replace('\s+', '-', regex=True)
        return df

    single_locus_columns = [
        'session_id',
        'pathogen_allele_id',
        'pathogen_strain',
        'expression',
    ]
    single_locus_rows = df.multiple_mutation.isna()
    multi_locus_columns = [
        'session_id',
        'multiple_mutation',
        'pathogen_strain',
        'pathogen_id',
    ]
    multi_locus_rows = df.multiple_mutation.notna()

    df = add_ids(df, single_locus_columns, single_locus_rows)
    df = fill_multiple_mutation_ids(df)
    df = add_ids(df, multi_locus_columns, multi_locus_rows, append_ids=True)
    df = add_wt_host_genotype_ids(df)
    df = df.drop('pathogen_genotype_n', axis=1)
    return df


def add_metagenotype_ids(df):
    columns = ['session_id', 'pathogen_genotype_id', 'canto_host_genotype_id']
    df = df.sort_values(columns)
    metagenotype_nums = []
    has_host_id = df.host_id.notna()
    rows_df = df.loc[has_host_id, columns]
    i = 1
    for current_row, next_row in itertools.pairwise(rows_df.values):
        session_id, *props = current_row
        next_session_id, *next_props = next_row
        metagenotype_nums.append(i)
        if session_id != next_session_id:
            i = 1
        elif props != next_props:
            i += 1
    # Append the final row because iteration stops early
    metagenotype_nums.append(i)
    metagenotype_ss = pd.Series(metagenotype_nums, dtype='str', index=rows_df.index)
    session_ids = df.loc[has_host_id, 'session_id']
    df.loc[has_host_id, 'metagenotype_id'] = (
        session_ids + '-metagenotype-' + metagenotype_ss
    )
    return df


def add_filamentous_classifier_column(filamentous_classifier, df):
    return df.merge(
        filamentous_classifier,
        left_on='pathogen_id',
        right_index=True,
        how='left',
    )


def add_disease_term_ids(mapping, df):
    label_to_id_map = {}
    diseases = df.disease
    unique_diseases = diseases.drop_duplicates().values
    for disease in unique_diseases:
        if disease is np.nan:
            continue
        term_ids = []
        term_labels = disease.lower().replace(', ', '; ').split('; ')
        for term_label in term_labels:
            term_id = mapping.get(term_label)
            if term_id is not np.nan and term_id.startswith('PHIDO'):
                term_ids.append(term_id)
        id_string = (
            '; '.join(t for t in term_ids if t is not np.nan) if term_ids else np.nan
        )
        label_to_id_map[disease] = id_string

    disease_ids = []
    for disease in diseases.values:
        ids = label_to_id_map[disease] if disease is not np.nan else disease
        disease_ids.append(ids)

    df['disease_id'] = disease_ids
    return df


def get_tissue_ids(tissue_mapping, phi_df):
    lookup = tissue_mapping.to_dict()
    sep = '; '
    tissue_ids = (
        phi_df.tissue[lambda x: x.notna()]
        .str.split(sep)
        .explode()
        .map(lookup)
        .fillna('')
        .groupby(level=0)
        .agg(sep.join)
        .reindex_like(phi_df)
    )
    return tissue_ids


def get_wt_metagenotype_ids(all_feature_mapping, phi_df):
    metagenotype_mapping = {
        mutant_id: wt_id
        for mapping in all_feature_mapping.values()
        for mutant_id, wt_id in mapping['metagenotypes'].items()
    }
    return phi_df.metagenotype_id.map(metagenotype_mapping)


def get_disease_annotations(phi_df):
    def get_disease_df(phi_df):
        columns = [
            'session_id',
            'wt_metagenotype_id',
            'disease_id',
            'tissue',
            'tissue_id',
            'phi_ids',
            'curation_date',
            'pmid',
        ]
        rows = phi_df.disease_id.notna() & phi_df.wt_metagenotype_id.notna()
        disease_df = phi_df.loc[rows, columns].copy()
        disease_df = disease_df[~disease_df.duplicated()]
        columns_to_split = ['disease_id', 'tissue', 'tissue_id', 'phi_ids']
        for col in columns_to_split:
            disease_df[col] = disease_df[col].str.split('; ')
        return disease_df

    disease_df = get_disease_df(phi_df)
    disease_annotations = defaultdict(list)
    for row in disease_df.itertuples(index=False):
        session_id = row.session_id
        metagenotype = row.wt_metagenotype_id
        if row.tissue_id is np.nan:
            extension = []
        else:
            extension = [
                {
                    'rangeDisplayName': tissue,
                    'rangeType': 'Ontology',
                    'rangeValue': tissue_id,
                    'relation': 'infects_tissue',
                }
                for tissue, tissue_id in zip(row.tissue, row.tissue_id)
            ]
        for term in row.disease_id:
            annotation = {
                'checked': 'yes',
                'conditions': [],
                'creation_date': str(row.curation_date),
                'curator': {'community_curated': False},
                'evidence_code': '',
                'extension': extension,
                'figure': '',
                'metagenotype': metagenotype,
                'phi4_id': row.phi_ids,
                'publication': f"PMID:{row.pmid}",
                'status': 'new',
                'term': term,
                'type': 'disease_name',
            }
            disease_annotations[session_id].append(annotation)
    return disease_annotations


def get_canto_json_template(df):
    def populate_curation_sessions(df):
        return {
            session_id: {
                'alleles': {},
                'annotations': [],
                'genes': {},
                'genotypes': {},
                'metadata': {},
                'metagenotypes': {},
                'organisms': {},
                'publications': {},
            }
            for session_id in df.session_id.unique()
        }

    return {
        'curation_sessions': populate_curation_sessions(df),
        'schema_version': 1,
    }


def add_gene_objects(canto_json, phi_df):
    curation_sessions = canto_json['curation_sessions']
    columns = ['session_id', 'canto_pathogen_gene_id', 'pathogen_species', 'protein_id']
    id_columns = ['session_id', 'canto_pathogen_gene_id']
    column_values = phi_df[columns].drop_duplicates(id_columns).values
    for session_id, gene_id, organism, uniprot_id in column_values:
        curation_sessions[session_id]['genes'][gene_id] = {
            'organism': organism,
            'uniquename': uniprot_id,
        }


def add_allele_objects(canto_json, phi_df):
    def format_gene_name_and_synonym(allele_type, gene_name, synonym):
        # TODO: Look for other automatic naming cases
        type_suffixes = {
            'deletion': 'delta',
            'wild_type': '+',
        }
        for type_name, suffix in type_suffixes.items():
            if allele_type == type_name:
                if suffix not in gene_name:
                    gene_name += suffix
                if synonym is not np.nan and suffix not in synonym:
                    synonym += suffix
        return (gene_name, synonym)

    curation_sessions = canto_json['curation_sessions']
    columns = [
        'session_id',
        'pathogen_allele_id',
        'allele_type',
        'canto_pathogen_gene_id',
        'description',
        'gene',
        'name',
        'pathogen_gene_synonym',
    ]
    id_columns = ['session_id', 'pathogen_allele_id']
    rows = phi_df[columns].drop_duplicates(id_columns).itertuples(index=False)

    for row in rows:
        allele_id = row.pathogen_allele_id
        synonym = row.pathogen_gene_synonym
        allele_name = row.name
        if row.name is np.nan:
            allele_name, synonym = format_gene_name_and_synonym(
                row.allele_type, row.gene, synonym
            )
        synonym_list = [] if synonym is np.nan else [synonym]
        allele = {
            'allele_type': row.allele_type,
            'description': row.description,
            'gene': row.canto_pathogen_gene_id,
            'name': allele_name,
            'primary_identifier': allele_id,
            'synonyms': synonym_list,
        }
        # This is probably inefficient, but inserting keys at a set location is a pain
        if row.description is np.nan or not row.description:
            del allele['description']

        curation_sessions[row.session_id]['alleles'][allele_id] = allele


def get_curation_date_df(phi_df):
    h1 = timedelta(hours=1)
    h2 = timedelta(hours=2)
    h9 = timedelta(hours=9)
    dates = pd.to_datetime(phi_df.set_index('session_id').curation_date)
    dfg = dates.groupby('session_id')
    df = dfg.agg(['min', 'max']).rename(columns={'min': 'start', 'max': 'end'})

    df['created'] = df['start'] + h9
    df['accepted'] = df['created'] + h1
    df['curation_accepted'] = df['accepted']
    df['curation_in_progress'] = np.where(
        df.start == df.end,
        df.curation_accepted + h1,
        df.end + h9,
    )
    df['first_submitted'] = df['curation_accepted'] + h2
    df['first_approved'] = df['first_submitted'] + h2
    df['needs_approval'] = np.where(
        df.start == df.end,
        df.first_submitted,
        df.curation_in_progress + h2,
    )
    df['approval_in_progress'] = df['needs_approval'] + h1
    df['approved'] = df['approval_in_progress'] + h1
    df['annotation_status'] = df['approved']
    return df.astype(str)


def add_wild_type_genotype_objects(canto_json, phi_df):
    curation_sessions = canto_json['curation_sessions']
    columns = [
        'session_id',
        'canto_host_genotype_id',
        'host_strain',
        'host_id',
    ]
    id_columns = ['session_id', 'canto_host_genotype_id']
    values = (
        phi_df[columns]
        .drop_duplicates(id_columns)
        .dropna(subset='canto_host_genotype_id')
        .values
    )
    for session_id, genotype_id, strain, taxon_id in values:
        if pd.isna(taxon_id):
            continue
        curation_sessions[session_id]['genotypes'][genotype_id] = {
            'loci': [],
            'organism_strain': strain if pd.notna(strain) else 'Unknown strain',
            'organism_taxonid': int(taxon_id),
        }


def add_mutant_genotype_objects(canto_json, phi_df):
    def make_locus_object(allele_id, expression):
        locus_allele = {}
        if pd.notna(expression):
            locus_allele['expression'] = expression
        locus_allele['id'] = allele_id
        return [locus_allele]

    def get_genotype_values(phi_df):
        columns = [
            'session_id',
            'pathogen_id',
            'pathogen_strain',
            'pathogen_genotype_id',
            'pathogen_allele_id',
            'expression',
        ]
        sort_order = ['session_id', 'pathogen_genotype_id', 'pathogen_allele_id']
        id_columns = ['pathogen_genotype_id', 'pathogen_allele_id']
        return (
            phi_df[columns].sort_values(sort_order).drop_duplicates(id_columns).values
        )

    def make_first_genotype(row_values):
        _, taxon_id, strain, _, allele_id, expression = row_values[0]
        genotype = {
            'loci': [make_locus_object(allele_id, expression)],
            'organism_strain': strain if pd.notna(strain) else 'Unknown strain',
            'organism_taxonid': taxon_id,
        }
        return genotype

    curation_sessions = canto_json['curation_sessions']
    values = get_genotype_values(phi_df)
    genotype = make_first_genotype(values)

    for previous_row, current_row in itertools.pairwise(values):
        previous_session_id = previous_row[0]
        previous_genotype_id = previous_row[3]
        _, taxon_id, strain, genotype_id, allele_id, expression = current_row

        if genotype_id != previous_genotype_id:
            curation_sessions[previous_session_id]['genotypes'][
                previous_genotype_id
            ] = genotype

            genotype = {
                'loci': [],
                'organism_strain': strain if pd.notna(strain) else 'Unknown strain',
                'organism_taxonid': taxon_id,
            }

        locus = make_locus_object(allele_id, expression)
        genotype['loci'].append(locus)

    # Iteration misses the final genotype
    session_id = current_row[0]
    curation_sessions[session_id]['genotypes'][genotype_id] = genotype


def add_genotype_objects(canto_json, phi_df):
    add_mutant_genotype_objects(canto_json, phi_df)
    add_wild_type_genotype_objects(canto_json, phi_df)


def add_metagenotype_objects(canto_json, phi_df):
    curation_sessions = canto_json['curation_sessions']
    columns = [
        'session_id',
        'metagenotype_id',
        'pathogen_genotype_id',
        'canto_host_genotype_id',
    ]
    id_columns = ['session_id', 'metagenotype_id']
    column_values = (
        phi_df[columns]
        .drop_duplicates(id_columns)
        .dropna(subset='metagenotype_id')
        .values
    )
    for (
        session_id,
        metagenotype_id,
        pathogen_genotype_id,
        canto_host_genotype_id,
    ) in column_values:
        curation_sessions[session_id]['metagenotypes'][metagenotype_id] = {
            'pathogen_genotype': pathogen_genotype_id,
            'host_genotype': canto_host_genotype_id,
            'type': 'pathogen-host',
        }


def add_organism_objects(canto_json, phi_df):
    curation_sessions = canto_json['curation_sessions']
    organism_df = pd.concat(
        [
            phi_df[['session_id', 'pathogen_id', 'pathogen_species']].rename(
                columns={
                    'pathogen_id': 'taxon_id',
                    'pathogen_species': 'species',
                }
            ),
            phi_df[['session_id', 'host_id', 'host_species']].rename(
                columns={
                    'host_id': 'taxon_id',
                    'host_species': 'species',
                }
            ),
        ]
    )

    organism_df.taxon_id = organism_df.taxon_id.astype('Int64')
    organism_data = organism_df.drop_duplicates(['session_id', 'taxon_id']).values
    for session_id, taxon_id, species in organism_data:
        if pd.isna(taxon_id):
            # TODO: Filter out invalid taxon IDs before this point
            continue
        organisms = curation_sessions[session_id]['organisms']
        # Taxon IDs must be strings to match the PHI-Canto export
        organisms[str(taxon_id)] = {'full_name': species}


def add_publication_objects(canto_json, phi_df):
    curation_sessions = canto_json['curation_sessions']
    session_publications = phi_df[['session_id', 'pmid']].drop_duplicates().values
    for session_id, pmid in session_publications:
        pmid_fmt = f"PMID:{pmid}"
        curation_sessions[session_id]['publications'][pmid_fmt] = {}


def make_phenotype_mapping(phenotype_mapping_df, phipo_mapping, phi_df):
    def split_mapping_by_column(phenotype_mapping_df):
        # Split the mapping table into multiple tables, one per column to be
        # mapped, into a dict keyed by the mapped column name.
        mapping_tables = {}
        columns = phenotype_mapping_df.columns
        is_mapping_name = columns.str.startswith('column_')
        is_mapping_value = columns.str.startswith('value_')
        mapping_columns = columns[is_mapping_name]
        mapping_values = columns[is_mapping_value]
        columns_to_exclude = [
            *mapping_columns,
            *mapping_values,
            'primary_label',
            'eco_label',
        ]
        columns_to_keep = columns.difference(columns_to_exclude, sort=False)
        dfg = phenotype_mapping_df.groupby('column_1')
        for column_name, gdf in dfg:
            columns_to_ignore = [
                col for col in mapping_columns if gdf[col].isna().all()
            ]
            columns_to_rename = mapping_columns.difference(columns_to_ignore)
            for col in mapping_values:
                gdf[col] = gdf[col].replace({'TRUE': True, 'FALSE': False})
            renames = {
                value_column: gdf[column].iloc[0]
                for column, value_column in zip(columns_to_rename, mapping_values)
            }
            index_columns = list(renames.values())
            df = gdf.rename(columns=renames).set_index(index_columns)[columns_to_keep]
            mapping_tables[column_name] = df
        return mapping_tables

    def index_mapping_by_record_id(mapping_tables, phi_df):
        # Change the phenotype mapping to map to row indexes, to make lookups
        # easier when iterating through the PHI-base DataFrame.
        indexed_mapping = {}
        for column_name, column_df in mapping_tables.items():
            index_columns = column_df.index.names
            df = (
                phi_df[['record_id'] + index_columns]
                .set_index(index_columns)
                .join(column_df)
                .set_index('record_id')
                .loc[lambda df: df.primary_id.notna() | df.extension_range.notna()]
            )
            indexed_mapping[column_name] = df
        return indexed_mapping

    def combine_column_mapping(phipo_mapping, indexed_mapping):
        # Convert the column mapping DataFrames into a single mapping dict,
        # keyed by row index of the PHI-base DataFrame.
        mapping_dict = defaultdict(list)
        for column_name, column_df in indexed_mapping.items():
            for row in column_df.itertuples():
                index = row.Index
                term = (
                    row.primary_id if row.primary_id is not np.nan else 'PHIPO:0000001'
                )
                data = {
                    'conditions': [],
                    'term': term,
                    'extension': [],
                    'feature_type': row.feature,
                }
                if row.eco_id is not np.nan:
                    data['conditions'].append(row.eco_id)
                has_extension = (
                    row.extension_range is not np.nan
                    and row.extension_relation is not np.nan
                )
                if has_extension:
                    # TODO: Handle range types other than 'Ontology' and ontologies
                    # other than PHIPO.
                    term_id = row.extension_range
                    label = phipo_mapping[term_id]
                    extension = {
                        'rangeDisplayName': label,
                        'rangeType': 'Ontology',
                        'rangeValue': term_id,
                        'relation': row.extension_relation,
                    }
                    data['extension'].append(extension)
                mapping_dict[index].append(data)
        return mapping_dict

    def merge_annotation_extensions(mapping_dict):
        def has_many_unique(annotations):
            unique = set()
            for annotation in annotations:
                extension_ranges = (
                    extension['rangeValue'] for extension in annotation['extension']
                )
                key = (annotation['term'], *extension_ranges)
                unique.add(key)
            return len(unique) > 1

        # Merge mutant phenotype annotation extensions
        for k, annotations in mapping_dict.items():
            if len(annotations) < 2:
                continue  # nothing to merge with one or fewer annotations
            specific_metagenotype_annotations = (
                annotation['feature_type'] == 'metagenotype'
                and annotation['term'] != 'PHIPO:0000001'
                for annotation in annotations
            )
            if not any(specific_metagenotype_annotations):
                # we have to merge onto a more specific interaction phenotype since
                # the infective_ability extension is invalid for pathogen phenotypes
                continue
            annotations_to_merge = [
                annotation
                for annotation in annotations
                if annotation['feature_type'] == 'metagenotype'
                and annotation['term'] == 'PHIPO:0000001'
                for extension in annotation['extension']
                if extension['relation'] == 'infective_ability'
            ]
            if not annotations_to_merge:
                continue
            if has_many_unique(annotations_to_merge):
                raise NotImplementedError(f'{k}: multiple annotations to merge')
            annotation_to_merge = annotations_to_merge[0]
            if len(annotation_to_merge['extension']) > 1:
                raise NotImplementedError(f'{k}: multiple extensions to merge')
            extension_to_merge = annotation_to_merge['extension'][0]

            annotations_to_merge_into = [
                annotation
                for annotation in annotations
                if annotation['feature_type'] == 'metagenotype'
                and (
                    not annotation['extension']
                    or all(
                        extension['relation'] != 'infective_ability'
                        for extension in annotation['extension']
                    )
                )
            ]
            for annotation in annotations_to_merge_into:
                annotation['extension'].append(extension_to_merge)

            # Exclude annotations that have been merged
            mapping_dict[k] = [
                d for d in mapping_dict[k] if d['term'] != 'PHIPO:0000001'
            ]

    mapping_dict = combine_column_mapping(
        phipo_mapping,
        index_mapping_by_record_id(
            split_mapping_by_column(phenotype_mapping_df), phi_df
        ),
    )
    merge_annotation_extensions(mapping_dict)
    return mapping_dict


def add_phenotype_annotations(canto_json, phenotype_lookup, phi_df):
    curation_sessions = canto_json['curation_sessions']
    for row in phi_df.itertuples():
        annotation_data = phenotype_lookup[row.record_id]
        if not annotation_data:
            continue

        session_id = row.session_id
        annotation_template = {
            'checked': 'yes',
            'conditions': [],
            'creation_date': str(row.curation_date),
            'evidence_code': 'Unknown',  # TODO: Add evidence codes
            'extension': [],
            'curator': {'community_curated': False},
            'figure': '',
            'phi4_id': row.phi_ids.split('; '),
            'publication': 'PMID:{}'.format(row.pmid),
            'status': 'new',
            'term': None,
            'type': 'pathogen_host_interaction_phenotype',
        }
        if row.comments is not np.nan:
            annotation_template['submitter_comment'] = row.comments

        if row.metagenotype_id is not np.nan:
            metagenotype_annotations = [
                annotation
                for annotation in annotation_data
                if annotation['feature_type'] == 'metagenotype'
            ]
            for annotation in metagenotype_annotations:
                phi_annotation = annotation_template.copy()
                phi_annotation['term'] = annotation['term']
                phi_annotation['metagenotype'] = row.metagenotype_id
                phi_annotation['conditions'] = annotation['conditions']
                extensions = annotation['extension']
                if row.tissue_id is not np.nan:
                    tissue_extensions = [
                        {
                            'rangeDisplayName': tissue,
                            'rangeType': 'Ontology',
                            'rangeValue': tissue_id,
                            'relation': 'infects_tissue',
                        }
                        for tissue, tissue_id in zip(
                            row.tissue.split('; '), row.tissue_id.split('; ')
                        )
                    ]
                    extensions.extend(tissue_extensions)
                phi_annotation['extension'] = extensions
                curation_sessions[session_id]['annotations'].append(phi_annotation)

        pathogen_annotations = [
            annotation
            for annotation in annotation_data
            if annotation['feature_type'] == 'pathogen_genotype'
        ]
        for annotation in pathogen_annotations:
            pathogen_annotation = annotation_template.copy()
            pathogen_annotation['type'] = 'pathogen_phenotype'
            pathogen_annotation['term'] = annotation['term']
            pathogen_annotation['genotype'] = row.pathogen_genotype_id
            pathogen_annotation['conditions'] = annotation['conditions']
            pathogen_annotation['extension'] = annotation['extension']
            curation_sessions[session_id]['annotations'].append(pathogen_annotation)


def add_wt_features(all_feature_mapping, canto_json):
    curation_sessions = canto_json['curation_sessions']
    for session_id, feature_mapping in all_feature_mapping.items():
        session = curation_sessions[session_id]
        wt_features = get_wt_features(feature_mapping, session)
        for feature_key, features in wt_features.items():
            for feature_id, feature in features.items():
                session[feature_key][feature_id] = feature


def add_disease_annotations(canto_json, phi_df):
    curation_sessions = canto_json['curation_sessions']
    disease_annotations = get_disease_annotations(phi_df)
    for session_id, annotations in disease_annotations.items():
        session = curation_sessions[session_id]
        session['annotations'].extend(annotations)


def add_annotation_objects(canto_json, phenotype_lookup, phi_df):
    add_phenotype_annotations(canto_json, phenotype_lookup, phi_df)
    add_disease_annotations(canto_json, phi_df)


def add_metadata_objects(canto_json, phi_df):
    date_df = get_curation_date_df(phi_df)
    dfg = phi_df.groupby('session_id')
    gene_counts = dfg.protein_id.nunique().astype(str).to_dict()
    pmids = ('PMID:' + dfg.pmid.first().astype(str)).to_dict()
    curation_sessions = canto_json['curation_sessions']
    for row in date_df.itertuples():
        session_id = row.Index
        session = curation_sessions[session_id]
        session['metadata'] = {
            'accepted_timestamp': row.accepted,
            'annotation_mode': 'standard',
            'annotation_status': 'APPROVED',
            'annotation_status_datestamp': row.annotation_status,
            'approval_in_progress_timestamp': row.approval_in_progress,
            'approved_timestamp': row.approved,
            'canto_session': session_id,
            'curation_accepted_date': row.curation_accepted,
            'curation_in_progress_timestamp': row.curation_in_progress,
            'curation_pub_id': pmids[session_id],
            'curator_role': 'Molecular Connections',
            'first_approved_timestamp': row.first_approved,
            'has_community_curation': False,
            'needs_approval_timestamp': row.needs_approval,
            'session_created_timestamp': row.created,
            'session_first_submitted_timestamp': row.first_submitted,
            'session_genes_count': gene_counts[session_id],
            'session_term_suggestions_count': '0',
            'session_unknown_conditions_count': '0',
            'term_suggestion_count': '0',
            'unknown_conditions_count': '0',
        }


def add_organism_roles(canto_json):
    pathogen_taxids, host_taxids = (
        set(pd.read_csv(path)['ncbi_taxid'])
        for path in (
            DATA_DIR / 'phicanto_pathogen_species.csv',
            DATA_DIR / 'phicanto_host_species.csv',
        )
    )
    for session in canto_json['curation_sessions'].values():
        if 'organisms' not in session:
            continue
        for taxon_id, organism in session['organisms'].items():
            taxid = int(taxon_id)
            organism['role'] = (
                'pathogen'
                if taxid in pathogen_taxids
                else 'host'
                if taxid in host_taxids
                else 'unknown'
            )


def add_pathogen_gene_names_to_alleles(canto_json, phi_df):
    gene_lookup = phi_df.set_index('protein_id')['gene'].to_dict()
    for session in canto_json['curation_sessions'].values():
        genes = session['genes']
        for allele in session['alleles'].values():
            gene = genes[allele['gene']]
            uniprot_id = gene['uniquename']
            gene_name = gene_lookup[uniprot_id]
            allele['gene_name'] = gene_name


def get_phi_id_column(phi_df):
    sep = '; '
    sort_uniques = lambda x: x if x is np.nan else sorted(list(set(x)))
    phi_ids = (
        phi_df.phi_molconn_id
        .str.cat(phi_df.multiple_mutation, sep)
        .mask(phi_df.multiple_mutation.isna(), phi_df.phi_molconn_id)
        .str.split(sep)
        .apply(sort_uniques)
        .str.join(sep)
    )
    return phi_ids


def make_combined_export(phibase_path, phicanto_path):
    phenotype_mapping_df = load_phenotype_column_mapping(
        DATA_DIR / 'phenotype_mapping.csv'
    )
    in_vitro_growth_classifier = load_in_vitro_growth_classifier(
        DATA_DIR / 'in_vitro_growth_mapping.csv'
    )
    disease_mapping = load_disease_column_mapping(
        phido_path=DATA_DIR / 'phido.csv',
        extra_path=DATA_DIR / 'disease_mapping.csv',
    )
    exp_tech_df = load_exp_tech_mapping(DATA_DIR / 'allele_mapping.csv')
    bto_mapping = load_bto_id_mapping(DATA_DIR / 'bto.csv')
    phipo_mapping = load_phipo_mapping(DATA_DIR / 'phipo.csv')

    canto_export = load_json(phicanto_path)

    approved_pmids = get_approved_pmids(canto_export)

    phi_df = clean_phibase_csv(phibase_path)

    phi_df = filter_phi_df(phi_df, approved_pmids)
    phi_df = add_session_ids(phi_df)
    phi_df = add_gene_ids(phi_df)
    phi_df = add_alleles_and_evidence(exp_tech_df, phi_df)
    phi_df = split_compound_columns(phi_df)

    # Remove unmappable allele types
    phi_df = phi_df[phi_df.allele_type.notna()]

    phi_df = add_allele_ids(phi_df)
    phi_df = add_genotype_ids(phi_df)
    phi_df = add_metagenotype_ids(phi_df)

    # Extract pathogen gene synonyms
    phi_df['pathogen_gene_synonym'] = phi_df.gene.str.extract(
        r'\((.+?)\)', expand=False
    )
    phi_df['gene'] = phi_df.gene.str.replace(re.compile(r'\s*\(.+?\)'), '')

    canto_json = get_canto_json_template(phi_df)

    add_gene_objects(canto_json, phi_df)
    add_organism_objects(canto_json, phi_df)
    add_allele_objects(canto_json, phi_df)
    add_genotype_objects(canto_json, phi_df)
    add_metagenotype_objects(canto_json, phi_df)
    add_metadata_objects(canto_json, phi_df)
    add_publication_objects(canto_json, phi_df)

    # Prepare for adding annotations
    all_feature_mapping = get_all_feature_mappings(canto_json)
    phi_df = add_filamentous_classifier_column(in_vitro_growth_classifier, phi_df)
    phi_df = add_disease_term_ids(disease_mapping, phi_df)
    phi_df['phi_ids'] = get_phi_id_column(phi_df)
    phi_df['tissue_id'] = get_tissue_ids(bto_mapping, phi_df)
    phi_df['wt_metagenotype_id'] = get_wt_metagenotype_ids(all_feature_mapping, phi_df)
    phenotype_lookup = make_phenotype_mapping(
        phenotype_mapping_df, phipo_mapping, phi_df
    )

    # Needed for converting allele names to wild type
    add_pathogen_gene_names_to_alleles(canto_json, phi_df)

    add_wt_features(all_feature_mapping, canto_json)
    add_annotation_objects(canto_json, phenotype_lookup, phi_df)
    add_organism_roles(canto_json)

    return canto_json
