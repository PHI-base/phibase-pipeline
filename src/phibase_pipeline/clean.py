# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from phibase_pipeline import loaders


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
        .str.replace(leading_trailing_underscores, '', regex=True)
        .str.replace(slash_or_hyphen, '_', regex=True)
        .str.replace(not_word_nor_space, '', regex=True)
        .str.replace(whitespace, '_', regex=True)
        .str.replace('invitro', 'in_vitro')
    )
    return column_names


def fix_curation_dates(curation_dates):
    # First we need to fix inconsistent month-year and year-month dates
    # with two-digit years, else these will fail to parse.
    month_year_pattern = r'^([A-Z][a-z]{2})-(\d{2})$'
    year_month_pattern = r'^(\d{2})-([A-Z][a-z]{2})$'
    fixed_dates = (
        curation_dates.ffill()  # Use previous date in case of blank rows
        .astype(str)
        .str.strip()
        .str.replace(month_year_pattern, r'\1-20\2', regex=True)
        .str.replace(year_month_pattern, r'\2-20\1', regex=True)
    )
    has_serial_date = curation_dates.astype(str).str.match('^\d+$', na=False)
    serial_dates = curation_dates.loc[has_serial_date].astype(int)
    fixed_dates.loc[has_serial_date] = pd.to_datetime(
        serial_dates, unit='D', origin='1899-12-30'
    )
    converted_dates = pd.to_datetime(fixed_dates, format='mixed', dayfirst=True)
    return converted_dates


def clean_phibase_csv(phi_df):
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
    tissue_replacements = loaders.load_tissue_replacements()
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
        .replace(column_replacements)
    )
    # This replacement raises a ValueError if either of the aa_sequence or
    # nt_sequence columns are empty (no idea why), so we need an extra check.
    for col in phi_df.columns:
        if phi_df[col].isna().all():
            continue
        phi_df[col] = phi_df[col].replace(ligatures, regex=True)

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

    phi_df.pathogen_id = phi_df.pathogen_id.astype('UInt64')

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
    phi_df.multiple_mutation = (
        phi_df.multiple_mutation
        .astype('object')
        .str.rstrip(';')
        .str.replace(separator_without_space, '; ', regex=True)
        .str.replace(space_separator, '; ', regex=True)
    )
    # Fix separators in disease column
    phi_df.disease = phi_df.disease.str.replace(',', ';')

    return phi_df


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
        phi_df = loaders.load_phibase_cleaned(cleaned_path)
    return phi_df
