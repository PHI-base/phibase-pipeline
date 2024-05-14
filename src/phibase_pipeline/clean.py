import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / 'data'


def get_normalized_column_names(phi_df):
    slash_or_hyphen = re.compile(r'[/-]')
    not_word_nor_space = re.compile(r'[^\w ]')
    leading_trailing_underscores = re.compile(r'^_+|_+$')
    whitespace = re.compile(r'\s+')
    column_names = (
        phi_df.columns
        .str.strip()
        .str.lower()
        .str.replace(leading_trailing_underscores, '')
        .str.replace(slash_or_hyphen, '_')
        .str.replace(not_word_nor_space, '')
        .str.replace(whitespace, '_')
        .str.replace('invitro', 'in_vitro')
    )
    return column_names


def load_tissue_replacements(path):
    return pd.read_csv(path, index_col='value')['rename'].dropna().to_dict()


def fix_curation_dates(curation_dates):
    def get_fixed_numeric_dates(dates):
        # Convert Excel numeric dates
        has_numeric_date = curation_dates.str.match('^\d+$', na=False)
        numeric_dates = curation_dates.loc[has_numeric_date].astype(int)
        converted_dates = pd.to_datetime(numeric_dates, unit='D', origin='1899-12-30')
        return converted_dates

    def get_fixed_month_day_dates(dates):
        # Convert month-day dates to day-month dates first
        month_day_pattern = re.compile(r'^([A-Z][a-z]{2})-(\d+)$')
        has_month_day_date = dates.str.match(month_day_pattern, na=False)
        month_day_dates = dates.loc[has_month_day_date]
        fixed_dates = month_day_dates.str.replace(month_day_pattern, r'\2-\1')
        return fixed_dates

    def get_fixed_day_month_dates(dates, numeric_dates):
        day_month_pattern = re.compile(r'^\d+-[A-Z][a-z]{2}$')
        has_day_month_dates = dates.str.match(day_month_pattern, na=False)
        # Forward-fill years from numeric dates (assumes dates are sorted)
        day_month_date_years = (
            numeric_dates.dt.year.astype(str)
            .reindex_like(dates)
            .fillna(method='ffill')
            .loc[has_day_month_dates]
        )
        day_month_dates = dates.loc[has_day_month_dates]
        day_month_year_dates = day_month_dates.str.cat(day_month_date_years, sep='-')
        fixed_dates = pd.to_datetime(day_month_year_dates, format='%d-%b-%Y')
        return fixed_dates

    curation_dates = curation_dates.copy()
    numeric_dates = get_fixed_numeric_dates(curation_dates)
    curation_dates.update(numeric_dates)
    curation_dates.update(get_fixed_month_day_dates(curation_dates))
    curation_dates.update(get_fixed_day_month_dates(curation_dates, numeric_dates))
    curation_dates = pd.to_datetime(curation_dates)
    return curation_dates


def clean_phibase_csv(path):
    phi_df = pd.read_csv(
        path,
        skipinitialspace=True,
        skiprows=1,
        na_values=['', 'no data found'],
        dtype=str,
    )
    phi_df.columns = get_normalized_column_names(phi_df)
    column_replacements = {
        'protein_id_source': {
            'Uniprot': 'UniProt',
        },
        'gene_id_source': {
            'Genbank': 'GenBank',
        },
        'host_id': {
            'No host tests done': np.nan,
            'Lethal pathogen phenotype': np.nan,
        },
        'ref_source': {
            'Pubmed': 'PubMed',
            'pubmed': 'PubMed',
        },
    }
    tissue_replacements = load_tissue_replacements(DATA_DIR / 'bto_renames.csv')
    column_replacements['tissue'] = tissue_replacements
    ligatures = {
        '\N{LATIN SMALL LIGATURE FF}': 'ff',
        '\N{LATIN SMALL LIGATURE FFI}': 'ffi',
        '\N{LATIN SMALL LIGATURE FFL}': 'ffl',
        '\N{LATIN SMALL LIGATURE FL}': 'fl',
        '\N{LATIN SMALL LIGATURE FI}': 'fi',
    }
    phi_df = (
        phi_df
        .replace('\N{NO-BREAK SPACE}', ' ', regex=True)
        .replace(r'^\s+|\s+$', '', regex=True)
        .replace('', np.nan)
        .replace('no data found', np.nan)
        .replace(ligatures, regex=True)
        .replace(column_replacements)
    )
    phi_df.curator_organization = (
        phi_df.curator_organization
        .replace(['Rres', 'rres', 'RRES'], 'RRes')
        .replace('MC/ Rres', 'MC/RRes')
    )
    prefix_id_columns = [
        'interacting_partners_id',
        'host_genotype_id',
        'host_target_id',
    ]
    phi_df[prefix_id_columns] = phi_df[prefix_id_columns].replace(
        ['Uniprot', 'Genbank'],
        ['UniProt', 'GenBank'],
        regex=True,
    )
    categorical_columns = [
        'mating_defect',
        'prepenetration_defect',
        'penetration_defect',
        'postpenetration_defect',
        'essential_gene',
    ]
    phi_df[categorical_columns] = phi_df[categorical_columns].replace(
        ['Yes', 'No'],
        ['yes', 'no'],
        regex=True,
    )
    # Remove common names from host species
    phi_df.host_species = phi_df.host_species.str.replace(r'\s*\(.+$', '', regex=True)
    lowercase_columns = [
        'vegetative_spores',
        'sexual_spores',
        'in_vitro_growth',
        'spore_germination',
        'exp_technique_stable',
    ]
    for col in lowercase_columns:
        column = phi_df[col]
        if pd.api.types.is_string_dtype(column):
            phi_df[col] = column.str.lower()

    phi_df.pathogen_id = phi_df.pathogen_id.astype('int64')

    nan_integer_columns = ['pathogen_strain_id', 'host_id', 'pmid', 'year']
    for col in nan_integer_columns:
        phi_df[col] = pd.to_numeric(phi_df[col], errors='coerce').astype('UInt64')

    phi_df = phi_df.rename(
        columns={
            'unnamed_66': 'lab_author',
            'unnamed_67': 'fg_mycotoxin',
            'unnamed_68': 'additional_gene_locus_id_source',
            'unnamed_69': 'additional_gene_locus_id',
            'unnamed_70': 'anti_infective_agent',
            'unnamed_71': 'anti_infective_compound',
            'unnamed_72': 'anti_infective_target_site',
            'unnamed_73': 'anti_infective_group_name',
            'unnamed_74': 'anti_infective_chemical_group',
            'unnamed_75': 'anti_infective_mode_in_planta',
            'unnamed_76': 'mode_of_action',
            'unnamed_77': 'frac_code',
            'unnamed_78': 'anti_infective_comments',
            'unnamed_79': 'cellular_phenotypes',
            'unnamed_80': 'horizontal_gene_transfer',
            'unnamed_81': 'phi3_pars',
            'unnamed_82': 'pathogen_gene_maps',
            'unnamed_83': 'host_gene_maps',
        }
    )
    phi_df.curation_date = fix_curation_dates(phi_df.curation_date)

    # Fix separators in multiple mutation column
    separator_without_space = re.compile(r';(?! )')
    space_separator = re.compile(r'(?<=\d) (?=PHI:)')
    if pd.api.types.is_string_dtype(phi_df.multiple_mutation):
        phi_df.multiple_mutation = (
            phi_df.multiple_mutation
            .str.rstrip(';')
            .str.replace(separator_without_space, '; ')
            .str.replace(space_separator, '; ')
        )
    # Fix separators in disease column
    phi_df.disease = phi_df.disease.str.replace(',', ';')

    return phi_df


def load_phibase_cleaned(path):
    dtypes = (
        pd.read_csv(DATA_DIR / 'cleaned_dtypes.csv', index_col='column')
        .squeeze('columns')
        .to_dict()
    )
    return pd.read_csv(path, dtype=dtypes)


def load_or_create_cleaned_csv(base_path, cleaned_path, force=False):
    """Load cleaned CSV if it exists, otherwise create it."""
    if force or not os.path.exists(cleaned_path):
        phi_df = clean_phibase_csv(base_path)
        phi_df.to_csv(
            cleaned_path,
            index=False,
            encoding='utf8',
            quoting=csv.QUOTE_NONNUMERIC,
        )
    else:
        phi_df = load_phibase_cleaned(cleaned_path)
    return phi_df
