import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from phibase_pipeline.ensembl import (
    combine_canto_uniprot_data,
    get_canto_columns,
    get_genotype_data,
    get_metagenotype_data,
    get_uniprot_columns,
)


TEST_DATA_DIR = Path(__file__).parent / 'data' / 'ensembl'

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
    assert expected == actual


def test_get_metagenotype_data(phicanto_export):
    session = phicanto_export['curation_sessions']['cc6cf06675cc6e13']
    metagenotype_id = 'cc6cf06675cc6e13-metagenotype-1'
    expected = {
        'uniprot_a': 'A0A0L0V3D9',
        'organism_a': 'Puccinia striiformis',
        'strain_a': 'CYR32',
        'modification_a': (
            'Pst_12806deltaSP(1-23) (partial amino acid deletion) [Not assayed]'
        ),
        'uniprot_b': None,
        'organism_b': 'Nicotiana benthamiana',
        'strain_b': 'Unknown strain',
        'modification_b': None,
    }
    actual = get_metagenotype_data(session, metagenotype_id)
    assert expected == actual


def test_get_canto_columns(phicanto_export):
    expected = pd.read_csv(TEST_DATA_DIR / 'get_canto_columns_expected.csv')
    actual = get_canto_columns(phicanto_export)
    assert_frame_equal(expected, actual)


def test_get_uniprot_columns():
    expected = pd.read_csv(
        TEST_DATA_DIR / 'get_uniprot_columns_expected.csv',
        dtype={
            'taxid_species': 'Int64',
            'taxid_strain': 'Int64',
        }
    )
    uniprot_data = pd.read_csv(TEST_DATA_DIR / 'uniprot_test_data.csv')
    actual = get_uniprot_columns(uniprot_data)
    assert_frame_equal(expected, actual)


def test_combine_canto_uniprot_data():
    canto_data = pd.read_csv(TEST_DATA_DIR / 'get_canto_columns_expected.csv')
    uniprot_data = pd.read_csv(TEST_DATA_DIR / 'get_uniprot_columns_expected.csv')
    expected = pd.read_csv(
        TEST_DATA_DIR / 'combine_canto_uniprot_data_expected.csv',
        dtype={
            'phibase_id': 'object',
            'ensembl_a': 'object',
            'taxid_strain_a': 'Int64',
            'taxid_strain_b': 'Int64',
            'high_level_terms': 'object',
        }
    )
    actual = combine_canto_uniprot_data(canto_data, uniprot_data)
    assert_frame_equal(expected, actual)
