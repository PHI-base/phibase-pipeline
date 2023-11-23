from pathlib import Path

import pandas as pd

from phibase_pipeline.migrate import (
    load_bto_id_mapping,
    load_phipo_mapping,
)

DATA_DIR = Path(__file__).parent / 'data'


def test_load_bto_id_mapping():
    data = {
        'tissues, cell types and enzyme sources': 'BTO:0000000',
        'culture condition:-induced cell': 'BTO:0000001',
        'culture condition:1,4-dichlorobenzene-grown cell': 'BTO:0000002',
    }
    expected = pd.Series(data, name='term_id').rename_axis('term_label')
    actual = load_bto_id_mapping(DATA_DIR / 'bto.csv')
    assert actual.equals(expected)


def test_load_phipo_mapping():
    expected = {
        'PHIPO:0000001': 'pathogen host interaction phenotype',
        'PHIPO:0000002': 'single species phenotype',
        'PHIPO:0000003': 'tissue phenotype',
        'PHIPO:0000004': 'unaffected pathogenicity',
    }
    actual = load_phipo_mapping(DATA_DIR / 'phipo.csv')
    assert actual == expected
