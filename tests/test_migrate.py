from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phibase_pipeline.migrate import (
    get_approved_pmids,
    load_bto_id_mapping,
    load_disease_column_mapping,
    load_exp_tech_mapping,
    load_in_vitro_growth_classifier,
    load_phenotype_column_mapping,
    load_phipo_mapping,
    split_compound_columns,
    split_compound_rows,
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
    pd.testing.assert_series_equal(actual, expected)


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
    pd.testing.assert_frame_equal(actual, expected)


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


def test_load_disease_column_mapping():
    expected = {
        'american foulbrood': 'PHIDO:0000011',
        'angular leaf spot': 'PHIDO:0000012',
        'anthracnose': 'PHIDO:0000013',
        'anthracnose leaf spot': 'PHIDO:0000014',
        'anthracnose (cruciferae)': 'PHIDO:0000013',
        'anthracnose (cucumber)': 'PHIDO:0000013',
        'anthracnose (cucurbitaceae)': 'PHIDO:0000013',
    }
    actual = load_disease_column_mapping(
        phido_path=DATA_DIR / 'phido.csv',
        extra_path=DATA_DIR / 'disease_mapping.csv',
    )
    assert actual == expected


def test_load_in_vitro_growth_classifier():
    expected = pd.Series(
        data={
            28447: True,
            645: False,
        },
        name='is_filamentous',
    ).rename_axis('ncbi_taxid')
    actual = load_in_vitro_growth_classifier(DATA_DIR / 'in_vitro_growth_mapping.csv')
    pd.testing.assert_series_equal(actual, expected)


def test_split_compound_rows():
    df = pd.DataFrame(
        {
            'a': ['alpha', 'beta'],
            'b': ['charlie', 'delta'],
            'c': ['echo; foxtrot', 'golf'],
        },
        index=[7, 8],
    )
    expected = pd.DataFrame(
        {
            'a': ['alpha', 'alpha', 'beta'],
            'b': ['charlie', 'charlie', 'delta'],
            'c': ['echo', 'foxtrot', 'golf'],
        },
        index=[0, 1, 2],
    )
    actual = split_compound_rows(df, column='c', sep='; ')
    pd.testing.assert_frame_equal(actual, expected)


split_compound_columns_data = {
    'pathogen_strain': (
        # Input
        pd.DataFrame(
            {
                'pathogen_strain': ['A11; B11'],
                'host_strain': ['cv. Alpha'],
                'name': ['alpha'],
                'allele_type': ['deletion'],
                'description': ['echo'],
                'other': ['golf'],
            },
            index=[9],
        ),
        # Expected
        pd.DataFrame(
            {
                'pathogen_strain': ['A11', 'B11'],
                'host_strain': ['cv. Alpha', 'cv. Alpha'],
                'name': ['alpha', 'alpha'],
                'allele_type': ['deletion', 'deletion'],
                'description': ['echo', 'echo'],
                'other': ['golf', 'golf'],
            },
            index=[0, 1],
        ),
    ),
    'host_strain': (
        # Input
        pd.DataFrame(
            {
                'pathogen_strain': ['A11'],
                'host_strain': ['cv. Alpha; cv. Beta'],
                'name': ['alpha'],
                'allele_type': ['deletion'],
                'description': ['echo'],
                'other': ['golf'],
            },
            index=[9],
        ),
        # Expected
        pd.DataFrame(
            {
                'pathogen_strain': ['A11', 'A11'],
                'host_strain': ['cv. Alpha', 'cv. Beta'],
                'name': ['alpha', 'alpha'],
                'allele_type': ['deletion', 'deletion'],
                'description': ['echo', 'echo'],
                'other': ['golf', 'golf'],
            },
            index=[0, 1],
        ),
    ),
    'name': (
        # Input
        pd.DataFrame(
            {
                'pathogen_strain': ['A11'],
                'host_strain': ['cv. Alpha'],
                'name': ['alpha | beta'],
                'allele_type': ['deletion'],
                'description': ['echo'],
                'other': ['golf'],
            },
            index=[9],
        ),
        # Expected
        pd.DataFrame(
            {
                'pathogen_strain': ['A11', 'A11'],
                'host_strain': ['cv. Alpha', 'cv. Alpha'],
                'name': ['alpha', 'beta'],
                'allele_type': ['deletion', 'deletion'],
                'description': ['echo', 'echo'],
                'other': ['golf', 'golf'],
            },
            index=[0, 1],
        ),
    ),
    'allele_type': (
        # Input
        pd.DataFrame(
            {
                'pathogen_strain': ['A11'],
                'host_strain': ['cv. Alpha'],
                'name': ['alpha'],
                'allele_type': ['deletion | wild_type'],
                'description': ['echo'],
                'other': ['golf'],
            },
            index=[9],
        ),
        # Expected
        pd.DataFrame(
            {
                'pathogen_strain': ['A11', 'A11'],
                'host_strain': ['cv. Alpha', 'cv. Alpha'],
                'name': ['alpha', 'alpha'],
                'allele_type': ['deletion', 'wild_type'],
                'description': ['echo', 'echo'],
                'other': ['golf', 'golf'],
            },
            index=[0, 1],
        ),
    ),
    'description': (
        # Input
        pd.DataFrame(
            {
                'pathogen_strain': ['A11'],
                'host_strain': ['cv. Alpha'],
                'name': ['alpha'],
                'allele_type': ['deletion'],
                'description': ['echo | foxtrot'],
                'other': ['golf'],
            },
            index=[9],
        ),
        # Expected
        pd.DataFrame(
            {
                'pathogen_strain': ['A11', 'A11'],
                'host_strain': ['cv. Alpha', 'cv. Alpha'],
                'name': ['alpha', 'alpha'],
                'allele_type': ['deletion', 'deletion'],
                'description': ['echo', 'foxtrot'],
                'other': ['golf', 'golf'],
            },
            index=[0, 1],
        ),
    ),
    'two_fields': (
        # Input
        pd.DataFrame(
            {
                'pathogen_strain': ['A11; B11'],
                'host_strain': ['cv. Alpha'],
                'name': ['alpha'],
                'allele_type': ['deletion | wild_type'],
                'description': ['echo'],
                'other': ['golf'],
            },
            index=[9],
        ),
        # Expected
        pd.DataFrame(
            {
                'pathogen_strain': ['A11', 'A11', 'B11', 'B11'],
                'host_strain': ['cv. Alpha', 'cv. Alpha', 'cv. Alpha', 'cv. Alpha'],
                'name': ['alpha', 'alpha', 'alpha', 'alpha'],
                'allele_type': ['deletion', 'wild_type', 'deletion', 'wild_type'],
                'description': ['echo', 'echo', 'echo', 'echo'],
                'other': ['golf', 'golf', 'golf', 'golf'],
            },
            index=[0, 1, 2, 3],
        ),
    ),
}


@pytest.mark.parametrize(
    'df,expected',
    split_compound_columns_data.values(),
    ids=split_compound_columns_data.keys(),
)
def test_split_compound_columns(df, expected):
    actual = split_compound_columns(df)
    pd.testing.assert_frame_equal(actual, expected)


def test_get_approved_pmids():
    input = {
        'curation_sessions': {
            '0123456879abcdef': {
                'metadata': {'annotation_status': 'APPROVED'},
                'publications': {'PMID:123': {}},
            },
            '1123456879abcdef': {
                'metadata': {'annotation_status': 'APPROVED'},
                'publications': {'PMID:124': {}},
            },
            '2123456879abcdef': {
                'metadata': {'annotation_status': 'CURATION_IN_PROGRESS'},
                'publications': {'PMID:125': {}},
            },
        }
    }
    expected = {123, 124}
    actual = get_approved_pmids(input)
    assert actual == expected
