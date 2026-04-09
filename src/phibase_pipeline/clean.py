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


def clean_phibase_csv(phi_df: pd.DataFrame):
    tissue_replacements = loaders.load_tissue_replacements()
    phi_df['tissue'].replace(tissue_replacements)
    # TODO: Handle missing placeholders in compound columns
    phi_df = phi_df.replace(r'no\s+data\s+found', regex=True)
    # Remove common names from host species
    phi_df['host_species'] = phi_df.host_species.str.replace(r'\s*\(.+$', '', regex=True)

    # Fix pathogen species in NCBI Taxonomy ID column
    invalid_index = phi_df.pathogen_id == 'Valsa pyri'
    phi_df.loc[invalid_index, 'pathogen_id'] = 578113
    phi_df.loc[invalid_index, 'pathogen_species'] = 'Cytospora mali'

    # Fix separators in disease column
    phi_df.disease = phi_df.disease.str.replace(',', ';')
    # Replace an invalid PubMed ID
    phi_df.pmid = phi_df.pmid.replace(24056836, 23881098)

    # Fix separators in experimental technique column
    phi_df.exp_technique_stable = (
        phi_df.exp_technique_stable.str.replace(r'\s*;\s*', '; ', regex=True)
    )

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
