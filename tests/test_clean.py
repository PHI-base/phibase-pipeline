# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from phibase_pipeline import loaders
from phibase_pipeline.clean import clean_phibase_csv


TEST_DATA_DIR = Path(__file__).parent / 'data'


@pytest.fixture
def phi_df():
    csv_path = TEST_DATA_DIR / 'clean' / 'clean_phibase_csv.csv'
    return loaders.load_phibase_csv(csv_path)


@pytest.fixture
def phi_df_cleaned():
    csv_path = TEST_DATA_DIR / 'clean' / 'clean_phibase_csv_expected.csv'
    return loaders.load_phibase_csv(csv_path)


def test_clean_phibase_csv(phi_df, phi_df_cleaned):
    actual = clean_phibase_csv(phi_df)
    expected = phi_df_cleaned
    assert_frame_equal(actual, expected)
