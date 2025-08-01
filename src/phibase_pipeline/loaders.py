# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / 'data'
SCHEMA_DIR = Path(__file__).parent / 'schema'


def load_phig_uniprot_mapping(path=None):
    path = DATA_DIR / 'phig_uniprot_mapping.csv' if path is None else path
    return pd.read_csv(path, index_col='uniprot_id')['phig_id'].to_dict()


def load_phipo_chebi_mapping(path=None):
    path = DATA_DIR / 'phipo_chebi_mapping.csv' if path is None else path
    return (
        pd.read_csv(path, index_col='phipo_id')
        .rename(
            columns={
                'chebi_id': 'id',
                'chebi_label': 'label',
            }
        )
        .to_dict(orient='index')
    )


def load_in_vitro_growth_classifier(path=None):
    path = DATA_DIR / 'in_vitro_growth_mapping.csv' if path is None else path
    df = pd.read_csv(path)
    df.is_filamentous = df.is_filamentous.str.lower()
    classifier_column = (
        df.set_index('ncbi_taxid')
        .is_filamentous.loc[lambda x: x.isin(('yes', 'no'))]
        .map({'yes': True, 'no': False})
    )
    return classifier_column


def load_disease_column_mapping(path=None):
    path = DATA_DIR / 'disease_mapping.csv' if path is None else path
    df = pd.read_csv(path)
    mapping = df.set_index('term_label')['term_id'].to_dict()
    return mapping


def load_phenotype_column_mapping(path=None, exclude_unmapped=True):
    path = DATA_DIR / 'phenotype_mapping.csv' if path is None else path
    df = pd.read_csv(path, na_values=['?'], dtype='str')
    for column in df.columns:
        df[column] = df[column].str.strip()
    df.value_2 = df.value_2.replace({'TRUE': True, 'FALSE': False})
    if exclude_unmapped:
        has_primary_id = df.primary_id.notna()
        has_extension = df.extension_range.notna() & df.extension_relation.notna()
        df = df[has_primary_id | has_extension]
    return df


def load_bto_id_mapping(path=None):
    path = DATA_DIR / 'bto.csv' if path is None else path
    renames = {
        'LABEL': 'term_label',
        'ID': 'term_id',
    }
    return pd.read_csv(path).rename(columns=renames).set_index('term_label')['term_id']


def load_phipo_mapping(path=None):
    path = DATA_DIR / 'phipo.csv' if path is None else path
    return pd.read_csv(path).set_index('ID')['LABEL'].to_dict()


def load_json(path):
    with open(path, encoding='utf8') as json_file:
        return json.load(json_file)


def load_exp_tech_mapping(path=None):
    path = DATA_DIR / 'allele_mapping.csv' if path is None else path
    def join_rows(column):
        if column.size == 1:
            return column
        if column.isna().all():
            return np.nan
        else:
            return ' | '.join(column.dropna().unique())

    unique_columns = ['exp_technique_stable', 'gene_protein_modification']
    df = (
        pd.read_csv(path).groupby(unique_columns, dropna=False, sort=False).agg(join_rows)
    )
    valid_rows = (df != '?').all(axis=1)
    assert ~df.index.duplicated().any()
    return df[valid_rows].reset_index()


def load_chemical_data(path=None):
    path = DATA_DIR / 'chemical_data.csv' if path is None else path
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = df[col].str.strip()
    mapping = (
        df.dropna(subset='phipo_id')
        .set_index('phipo_id')
        .replace({np.nan: None})
        .to_dict(orient='index')
    )
    return mapping


def load_tissue_replacements(path=None):
    path = DATA_DIR / 'bto_renames.csv' if path is None else path
    return pd.read_csv(path, index_col='value')['rename'].dropna().to_dict()


def load_phibase_cleaned(path):
    dtypes = (
        pd.read_csv(DATA_DIR / 'cleaned_dtypes.csv', index_col='column')
        .squeeze('columns')
        .to_dict()
    )
    return pd.read_csv(path, dtype=dtypes)


def load_phibase5_json_schema():
    return load_json(SCHEMA_DIR / 'phibase5_import.schema.json')


def load_phibase_csv(path):
    return pd.read_csv(
        path,
        skipinitialspace=True,
        skiprows=1,
        na_values=['', 'no data found'],
        dtype=str,
    )


def load_obsolete_phido_mapping(path=None):
    path = DATA_DIR / 'obsolete_phido_mapping.csv' if path is None else path
    column_renames = {
        'term_id': 'term_id',
        'replaced_by_id': 'replaced_by',
        'consider_id': 'consider',
    }
    columns = list(column_renames)
    df = pd.read_csv(path)[columns].rename(columns=column_renames)
    df.replaced_by = df.replaced_by.mask(df.replaced_by.isna(), df.consider)
    rows = df.itertuples(index=False)
    term_groups = itertools.groupby(rows, key=lambda row: row.term_id)
    mapping = {}
    for obsolete_id, rows in term_groups:
        replacement_ids = sorted(
            list(set(row.replaced_by for row in rows if row.replaced_by is not np.nan))
        )
        mapping[obsolete_id] = replacement_ids
    return mapping
