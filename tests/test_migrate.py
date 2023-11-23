from pathlib import Path

import numpy as np
import pandas as pd

from phibase_pipeline.migrate import (
    load_bto_id_mapping,
    load_exp_tech_mapping,
    load_phenotype_column_mapping,
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


def test_load_exp_tech_mapping():
    data = [
        {
            "exp_technique_stable": "AA substitution with overexpression",
            "gene_protein_modification": "D144P-Y222A",
            "exp_technique_transient": np.nan,
            "allele_type": "amino acid substitution(s)",
            "name": np.nan,
            "description": "D144P-Y222A",
            "expression": "Overexpression",
            "evidence_code": "Unknown",
            "eco": np.nan,
        },
        {
            "exp_technique_stable": "AA substitution and deletion",
            "gene_protein_modification": "ABCdelta, D144A-Y222A",
            "exp_technique_transient": np.nan,
            "allele_type": "amino acid substitution(s) | deletion",
            "name": np.nan,
            "description": "D144A,Y222A",
            "expression": "Overexpression",
            "evidence_code": "Unknown",
            "eco": np.nan,
        },
        {
            "exp_technique_stable": "duplicate row",
            "gene_protein_modification": "D144A-Y222A",
            "exp_technique_transient": np.nan,
            "allele_type": "amino acid substitution(s)",
            "name": np.nan,
            "description": "D144A,Y222A",
            "expression": "Overexpression",
            "evidence_code": "Unknown",
            "eco": np.nan,
        },
    ]
    expected = pd.DataFrame.from_records(data)
    actual = load_exp_tech_mapping(DATA_DIR / 'allele_mapping.csv')
    assert actual.equals(expected), actual.compare(
        expected, result_names=('actual', 'expected')
    )


def test_load_phenotype_column_mapping():
    data = [
        {
            'column_1': 'in_vitro_growth',
            'value_1': 'increased',
            'column_2': 'is_filamentous',
            'value_2': 'FALSE',
            'primary_id': 'PHIPO:0000975',
            'primary_label': 'increased unicellular population growth',
            'eco_id': np.nan,
            'eco_label': np.nan,
            'feature': 'pathogen_genotype',
            'extension_relation': np.nan,
            'extension_range': np.nan,
        },
        {
            'column_1': 'mutant_phenotype',
            'value_1': 'enhanced antagonism',
            'column_2': np.nan,
            'value_2': np.nan,
            'primary_id': np.nan,
            'primary_label': np.nan,
            'eco_id': np.nan,
            'eco_label': np.nan,
            'feature': 'metagenotype',
            'extension_relation': 'infective_ability',
            'extension_range': 'PHIPO:0000207',
        },
    ]
    expected = pd.DataFrame(data, index=[0, 2], dtype='str')
    actual = load_phenotype_column_mapping(DATA_DIR / 'phenotype_mapping.csv')
    pd.testing.assert_frame_equal(actual, expected)
