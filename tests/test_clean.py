# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from phibase_pipeline import loaders
from phibase_pipeline.clean import clean_phibase_csv


TEST_DATA_DIR = Path(__file__).parent / 'data'


def test_clean_phibase_csv():
    csv_path = TEST_DATA_DIR / 'clean' / 'clean_phibase_csv.csv'
    expected_path = TEST_DATA_DIR / 'clean' / 'clean_phibase_csv_expected.csv'
    phi_df = loaders.load_phibase_csv(csv_path)
    actual = clean_phibase_csv(phi_df)
    expected = pd.read_csv(expected_path)
    # Fix a comparison error where actual dates have times, but expected ones don't.
    expected['curation_date'] = pd.to_datetime(expected['curation_date']).dt.normalize()
    expected = expected.astype(
        {
            'pathogen_id': 'UInt64',
            'pathogen_strain_id': 'UInt64',
            'host_id': 'UInt64',
            'pmid': 'UInt64',
            'year': 'UInt64',
        }
    )
    assert_frame_equal(actual, expected)
