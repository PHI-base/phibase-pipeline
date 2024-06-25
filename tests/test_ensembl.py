import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from phibase_pipeline.ensembl import (
    get_canto_columns,
    get_genotype_data,
)


TEST_DATA_DIR = Path(__file__).parent / 'data'

@pytest.fixture
def phicanto_export():
    path = TEST_DATA_DIR / 'phicanto_export.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def test_get_genotype_data(phicanto_export):
    session = phicanto_export['curation_sessions']['cc6cf06675cc6e13']
    genotype_id = 'cc6cf06675cc6e13-genotype-1'
    suffix = '_a'
    expected = {
        'uniprot_a': 'A0A0L0V3D9',
        'organism_a': 'Puccinia striiformis',
        'strain_a': 'CYR32',
        'modification_a': (
            'Pst_12806deltaSP(1-23) (partial amino acid deletion) [Not assayed]'
        )
    }
    actual = get_genotype_data(session, genotype_id, suffix)
    assert actual == expected
