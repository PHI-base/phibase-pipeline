from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phibase_pipeline.migrate import (
    add_allele_ids,
    add_allele_objects,
    add_annotation_objects,
    add_disease_annotations,
    add_disease_term_ids,
    add_filamentous_classifier_column,
    add_gene_ids,
    add_gene_objects,
    add_genotype_ids,
    add_genotype_objects,
    add_metadata_objects,
    add_metagenotype_ids,
    add_metagenotype_objects,
    add_mutant_genotype_objects,
    add_organism_objects,
    add_organism_roles,
    add_pathogen_gene_names_to_alleles,
    add_phenotype_annotations,
    add_publication_objects,
    add_session_ids,
    add_wild_type_genotype_objects,
    add_wt_features,
    fill_multiple_mutation_ids,
    get_approved_pmids,
    get_canto_json_template,
    get_curation_date_df,
    get_disease_annotations,
    get_phi_id_column,
    get_tissue_ids,
    get_wt_metagenotype_ids,
    load_bto_id_mapping,
    load_disease_column_mapping,
    load_exp_tech_mapping,
    load_in_vitro_growth_classifier,
    load_phenotype_column_mapping,
    load_phipo_mapping,
    make_combined_export,
    make_phenotype_mapping,
    make_phibase_json,
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
            'value_2': False,
            'primary_id': 'PHIPO:0000975',
            'primary_label': 'increased unicellular population growth',
            'eco_id': np.nan,
            'eco_label': np.nan,
            'feature': 'pathogen_genotype',
            'extension_relation': np.nan,
            'extension_range': np.nan,
        },
        {
            'column_1': 'in_vitro_growth',
            'value_1': 'increased',
            'column_2': 'is_filamentous',
            'value_2': True,
            'primary_id': 'PHIPO:0001234',
            'primary_label': 'increased hyphal growth',
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
    expected = pd.DataFrame(data, index=[0, 2, 3], dtype='object')
    actual = load_phenotype_column_mapping(DATA_DIR / 'phenotype_mapping.csv')
    pd.testing.assert_frame_equal(actual, expected)


def test_load_disease_column_mapping():
    expected = {
        # From PHIDO
        'american foulbrood': 'PHIDO:0000011',
        'angular leaf spot': 'PHIDO:0000012',
        'anthracnose': 'PHIDO:0000013',
        'anthracnose leaf spot': 'PHIDO:0000014',
        # From disease mapping
        'airsacculitis': 'PHIDO:0000008',
        'bacterial canker': 'PHIDO:0000025',
        'anthracnose (cruciferae)': 'PHIDO:0000013',
        'anthracnose (cucurbitaceae)': 'PHIDO:0000013',
        'anthracnose (cucumber)': 'PHIDO:0000013',
        'biocontrol: non pathogenic': np.nan,
        'blight': 'blight',
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


def test_add_session_ids():
    df = pd.DataFrame({'pmid': [123, 456, 789]})
    expected = pd.DataFrame(
        {
            'pmid': [
                123,
                456,
                789,
            ],
            'session_id': [
                'a665a45920422f9d',
                'b3a8e0e1f9ab1bfe',
                '35a9e381b1a27567',
            ],
        }
    )
    add_session_ids(df)
    pd.testing.assert_frame_equal(df, expected)


def test_add_gene_ids():
    df = pd.DataFrame(
        {
            'pathogen_species': ['Fusarium graminearum'],
            'protein_id': ['Q00909'],
            'host_genotype_id': ['UniProt: P43296'],
            'host_species': ['Arabidopsis thaliana'],
        }
    )
    expected = pd.DataFrame(
        {
            'pathogen_species': ['Fusarium graminearum'],
            'protein_id': ['Q00909'],
            'host_genotype_id': ['UniProt: P43296'],
            'host_species': ['Arabidopsis thaliana'],
            'canto_host_gene_id': ['Arabidopsis thaliana P43296'],
            'canto_pathogen_gene_id': ['Fusarium graminearum Q00909'],
        }
    )
    add_gene_ids(df)
    pd.testing.assert_frame_equal(df, expected)


test_add_allele_ids_data = {
    'identical_alleles': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
            ]
        ),
    ),
    'different_protein_id': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q9LUW0',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q9LUW0',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q9LUW0:0123456789abcdef-1',
                },
            ]
        ),
    ),
    'different_allele_type': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'wild_type',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'wild_type',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-2',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
            ]
        ),
    ),
    'different_name': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'wild_type',
                    'name': 'TRI5+',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'wild_type',
                    'name': 'TRI5+',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-2',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
            ]
        ),
    ),
    'different_description': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test B',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test B',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-2',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences should be ignored',
                    'ignore_b': 'differences should be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
            ]
        ),
    ),
    'different_ignored': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences could be ignored',
                    'ignore_b': 'differences might be ignored',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences may be ignored',
                    'ignore_b': 'differences must be ignored',
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences could be ignored',
                    'ignore_b': 'differences might be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
                {
                    'session_id': '0123456789abcdef',
                    'protein_id': 'Q00909',
                    'allele_type': 'deletion',
                    'name': 'TRI5delta',
                    'description': 'test A',
                    'ignore_a': 'differences may be ignored',
                    'ignore_b': 'differences must be ignored',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                },
            ]
        ),
    ),
}


@pytest.mark.parametrize(
    'df,expected',
    test_add_allele_ids_data.values(),
    ids=test_add_allele_ids_data.keys(),
)
def test_add_allele_ids(df, expected):
    actual = add_allele_ids(df.copy())
    pd.testing.assert_frame_equal(actual, expected)


def test_fill_multiple_mutation_ids():
    df = pd.DataFrame.from_records(
        [
            {
                'phi_molconn_id': 'PHI:1',
                'multiple_mutation': 'PHI:2',
            },
            {
                'phi_molconn_id': 'PHI:2',
                'multiple_mutation': 'PHI:1',
            },
            {
                'phi_molconn_id': 'PHI:1',
                'multiple_mutation': 'PHI:3; PHI:2',
            },
            {
                'phi_molconn_id': 'PHI:4',
                'multiple_mutation': np.nan,
            },
        ]
    )
    expected = pd.DataFrame.from_records(
        [
            {
                'phi_molconn_id': 'PHI:1',
                'multiple_mutation': 'PHI:1; PHI:2',
            },
            {
                'phi_molconn_id': 'PHI:2',
                'multiple_mutation': 'PHI:1; PHI:2',
            },
            {
                'phi_molconn_id': 'PHI:1',
                'multiple_mutation': 'PHI:1; PHI:2; PHI:3',
            },
            {
                'phi_molconn_id': 'PHI:4',
                'multiple_mutation': np.nan,
            },
        ]
    )
    actual = fill_multiple_mutation_ids(df.copy())
    pd.testing.assert_frame_equal(actual, expected)


test_add_genotype_ids_data = {
    'single_locus_diff_expression': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Knockdown',
                },
            ],
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-2',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_strain': 'PH-1',
                    'pathogen_id': 5518,
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Knockdown',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
            ],
        ),
    ),
    'single_locus_identical': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                },
            ],
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': np.nan,
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
            ],
        ),
    ),
    'multi_locus_unique': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:1',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:2',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                },
            ],
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:1; PHI:3',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:2; PHI:3',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_strain': 'PH-1',
                    'pathogen_id': 5518,
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-2',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
            ],
        ),
    ),
    'multi_locus_identical': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:1',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:1',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                },
            ],
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:1; PHI:3',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
                {
                    'phi_molconn_id': 'PHI:3',
                    'session_id': '0123456789abcdef',
                    'multiple_mutation': 'PHI:1; PHI:3',
                    'pathogen_allele_id': 'Q00909:0123456789abcdef-1',
                    'pathogen_id': 5518,
                    'pathogen_strain': 'PH-1',
                    'host_species': 'Triticum aestivum',
                    'host_id': 4565,
                    'host_strain': 'cv. Chinese Spring',
                    'expression': 'Overexpression',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                },
            ],
        ),
    ),
}


@pytest.mark.parametrize(
    'df,expected',
    test_add_genotype_ids_data.values(),
    ids=test_add_genotype_ids_data.keys(),
)
def test_add_genotype_ids(df, expected):
    actual = add_genotype_ids(df.copy())
    pd.testing.assert_frame_equal(actual, expected)


test_add_metagenotype_ids_data = {
    'empty_dataframe': pytest.param(
        pd.DataFrame(),
        None,
        marks=[pytest.mark.xfail],
    ),
    'single_metagenotype': (
        # Input
        pd.DataFrame(
            {
                'session_id': ['0123456789abcdef'],
                'pathogen_genotype_id': ['0123456789abcdef-genotype-1'],
                'canto_host_genotype_id': [
                    'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring'
                ],
                'host_id': [4565],
            }
        ),
        # Expected
        pd.DataFrame(
            {
                'session_id': ['0123456789abcdef'],
                'pathogen_genotype_id': ['0123456789abcdef-genotype-1'],
                'canto_host_genotype_id': [
                    'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring'
                ],
                'host_id': [4565],
                'metagenotype_id': ['0123456789abcdef-metagenotype-1'],
            }
        ),
    ),
    'identical_metagenotypes': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-1',
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-1',
                },
            ]
        ),
    ),
    'different_pathogen_genotypes': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-2',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-2',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-2',
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-1',
                },
            ]
        ),
    ),
    'different_host_genotypes': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Bobwhite',
                    'host_id': 4565,
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Bobwhite',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-1',
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-2',
                },
            ]
        ),
    ),
    'multiple_sessions': (
        # Input
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-2',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
                {
                    'session_id': '1123456789abcdef',
                    'pathogen_genotype_id': '1123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                },
            ]
        ),
        # Expected
        pd.DataFrame.from_records(
            [
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-1',
                },
                {
                    'session_id': '0123456789abcdef',
                    'pathogen_genotype_id': '0123456789abcdef-genotype-2',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '0123456789abcdef-metagenotype-2',
                },
                {
                    'session_id': '1123456789abcdef',
                    'pathogen_genotype_id': '1123456789abcdef-genotype-1',
                    'canto_host_genotype_id': 'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring',
                    'host_id': 4565,
                    'metagenotype_id': '1123456789abcdef-metagenotype-1',
                },
            ]
        ),
    ),
    'no_hosts': (
        # Input
        pd.DataFrame(
            {
                'session_id': ['0123456789abcdef'],
                'pathogen_genotype_id': ['0123456789abcdef-genotype-1'],
                'canto_host_genotype_id': [
                    'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring'
                ],
                'host_id': [np.nan],
            }
        ),
        # Expected
        pd.DataFrame(
            {
                'session_id': ['0123456789abcdef'],
                'pathogen_genotype_id': ['0123456789abcdef-genotype-1'],
                'canto_host_genotype_id': [
                    'Triticum-aestivum-wild-type-genotype-cv.-Chinese-Spring'
                ],
                'host_id': [np.nan],
                'metagenotype_id': [np.nan],
            }
        ),
    ),
}


@pytest.mark.parametrize(
    'df,expected',
    test_add_metagenotype_ids_data.values(),
    ids=test_add_metagenotype_ids_data.keys(),
)
def test_add_metagenotype_ids(df, expected):
    actual = add_metagenotype_ids(df.copy())
    pd.testing.assert_frame_equal(actual, expected)


def test_add_filamentous_classifier_column():
    df = pd.DataFrame(
        [
            ['Clavibacter michiganensis', 28447],
            ['Candida dubliniensis', 42374],
            ['Puccinia striiformis', 27350],
            ['Aeromonas salmonicida', 645],
        ],
        columns=['pathogen_species', 'pathogen_id'],
    )
    expected = pd.DataFrame(
        [
            ['Clavibacter michiganensis', 28447, True],
            ['Candida dubliniensis', 42374, np.nan],
            ['Puccinia striiformis', 27350, np.nan],
            ['Aeromonas salmonicida', 645, False],
        ],
        columns=['pathogen_species', 'pathogen_id', 'is_filamentous'],
    )
    filamentous_classifier = load_in_vitro_growth_classifier(
        DATA_DIR / 'in_vitro_growth_mapping.csv'
    )
    actual = add_filamentous_classifier_column(filamentous_classifier, df)
    pd.testing.assert_frame_equal(actual, expected)


def test_add_disease_term_ids():
    df = pd.DataFrame(
        [
            ['airsacculitis'],
            ['airsacculitis; bacterial canker'],
            ['anthracnose (Cruciferae)'],
            ['anthracnose (cucurbitaceae); anthracnose (cucumber)'],
            ['biocontrol: non pathogenic'],
            [np.nan],
        ],
        columns=['disease'],
    )
    expected = pd.DataFrame(
        [
            ['airsacculitis', 'PHIDO:0000008'],
            ['airsacculitis; bacterial canker', 'PHIDO:0000008; PHIDO:0000025'],
            ['anthracnose (Cruciferae)', 'PHIDO:0000013'],
            ['anthracnose (cucurbitaceae); anthracnose (cucumber)', 'PHIDO:0000013'],
            ['biocontrol: non pathogenic', np.nan],
            [np.nan, np.nan],
        ],
        columns=['disease', 'disease_id'],
    )
    mapping = load_disease_column_mapping(
        phido_path=DATA_DIR / 'phido.csv',
        extra_path=DATA_DIR / 'disease_mapping.csv',
    )
    actual = add_disease_term_ids(mapping, df)
    pd.testing.assert_frame_equal(actual, expected)


def test_get_tissue_ids():
    df = pd.DataFrame(
        [
            ['culture condition:-induced cell'],
            ['tissues, cell types and enzyme sources; culture condition:-induced cell'],
            ['invalid tissue'],
            [
                'culture condition:-induced cell; invalid tissue; tissues, cell types and enzyme sources'
            ],
        ],
        columns=['tissue'],
    )
    expected = pd.Series(
        [
            'BTO:0000001',
            'BTO:0000000; BTO:0000001',
            np.nan,
            'BTO:0000001; BTO:0000000',
        ],
        name='tissue',
    )
    tissue_mapping = load_bto_id_mapping(DATA_DIR / 'bto.csv')
    actual = get_tissue_ids(tissue_mapping, df)
    pd.testing.assert_series_equal(actual, expected)


def test_get_wt_metagenotype_ids():
    all_feature_mapping = {
        'f86e79b5ed64a342': {
            'alleles': {'P26215:f86e79b5ed64a342-1': 'P26215:f86e79b5ed64a342-2'},
            'genotypes': {'f86e79b5ed64a342-genotype-1': 'f86e79b5ed64a342-genotype-2'},
            'metagenotypes': {
                'f86e79b5ed64a342-metagenotype-1': 'f86e79b5ed64a342-metagenotype-2'
            },
        }
    }
    phi_df = pd.DataFrame(
        {
            'metagenotype_id': [
                'f86e79b5ed64a342-metagenotype-1',
                'f86e79b5ed64a342-metagenotype-3',
            ]
        }
    )
    actual = get_wt_metagenotype_ids(all_feature_mapping, phi_df)
    expected = pd.Series(
        [
            'f86e79b5ed64a342-metagenotype-2',
            np.nan,
        ],
        name='metagenotype_id',
    )
    pd.testing.assert_series_equal(actual, expected)


def test_get_disease_annotations():
    data = {
        13543: {
            'session_id': 'addf90acaa7ff81f',
            'wt_metagenotype_id': 'addf90acaa7ff81f-metagenotype-20',
            'disease_id': 'PHIDO:0000120',
            'tissue': 'kidney',
            'tissue_id': 'BTO:0000671',
            'phi_ids': 'PHI:10643; PHI:10644; PHI:10645',
            'curation_date': '2020-09-01 00:00:00',
            'pmid': 32612591,
        },
        # No tissue extensions
        11750: {
            'session_id': '3fce7a4d71754edb',
            'wt_metagenotype_id': '3fce7a4d71754edb-metagenotype-12',
            'disease_id': 'PHIDO:0000174',
            'tissue': np.nan,
            'tissue_id': np.nan,
            'phi_ids': 'PHI:9498',
            'curation_date': '2019-10-01 00:00:00',
            'pmid': 28754208,
        },
        # No metagenotype: don't convert
        6081: {
            'session_id': 'a188a67835808502',
            'wt_metagenotype_id': np.nan,
            'disease_id': 'PHIDO:0000065',
            'tissue': 'leaf',
            'tissue_id': 'BTO:0000713',
            'phi_ids': 'PHI:6106',
            'curation_date': '2016-04-01 00:00:00',
            'pmid': 26954255,
        },
        # No disease term ID: don't convert
        4637: {
            'session_id': '19561d4e39de7238',
            'wt_metagenotype_id': '19561d4e39de7238-metagenotype-33',
            'disease_id': np.nan,
            'tissue': np.nan,
            'tissue_id': np.nan,
            'phi_ids': 'PHI:4712',
            'curation_date': '2015-05-01 00:00:00',
            'pmid': 25845292,
        },
    }
    expected = {
        '3fce7a4d71754edb': [
            {
                'checked': 'yes',
                'conditions': [],
                'creation_date': '2019-10-01',
                'curator': {'community_curated': False},
                'evidence_code': '',
                'extension': [],
                'figure': '',
                'metagenotype': '3fce7a4d71754edb-metagenotype-12',
                'phi4_id': ['PHI:9498'],
                'publication': 'PMID:28754208',
                'status': 'new',
                'term': 'PHIDO:0000174',
                'type': 'disease_name',
            }
        ],
        'addf90acaa7ff81f': [
            {
                'checked': 'yes',
                'conditions': [],
                'creation_date': '2020-09-01',
                'curator': {'community_curated': False},
                'evidence_code': '',
                'extension': [
                    {
                        'rangeDisplayName': 'kidney',
                        'rangeType': 'Ontology',
                        'rangeValue': 'BTO:0000671',
                        'relation': 'infects_tissue',
                    }
                ],
                'figure': '',
                'metagenotype': 'addf90acaa7ff81f-metagenotype-20',
                'phi4_id': ['PHI:10643', 'PHI:10644', 'PHI:10645'],
                'publication': 'PMID:32612591',
                'status': 'new',
                'term': 'PHIDO:0000120',
                'type': 'disease_name',
            }
        ],
    }
    phi_df = pd.DataFrame.from_dict(data, orient='index')
    phi_df.curation_date = pd.to_datetime(phi_df.curation_date)
    actual = get_disease_annotations(phi_df)
    assert actual == expected


def test_get_canto_json_template():
    phi_df = pd.DataFrame(
        {
            'session_id': [
                '3a7f9b2e8c4d71e5',
                'b6e0a5f3d92c8b17',
            ]
        }
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'alleles': {},
                'annotations': [],
                'genes': {},
                'genotypes': {},
                'metadata': {},
                'metagenotypes': {},
                'organisms': {},
                'publications': {},
            },
            'b6e0a5f3d92c8b17': {
                'alleles': {},
                'annotations': [],
                'genes': {},
                'genotypes': {},
                'metadata': {},
                'metagenotypes': {},
                'organisms': {},
                'publications': {},
            },
        },
        'schema_version': 1,
    }
    actual = get_canto_json_template(phi_df)
    assert actual == expected


def test_add_gene_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'genes': {}},
            'b6e0a5f3d92c8b17': {'genes': {}},
        }
    }
    phi_df = pd.DataFrame(
        {
            'session_id': ['b6e0a5f3d92c8b17'],
            'canto_pathogen_gene_id': ['Escherichia coli P10486'],
            'pathogen_species': ['Escherichia coli'],
            'protein_id': ['P10486'],
        }
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'genes': {}},
            'b6e0a5f3d92c8b17': {
                'genes': {
                    'Escherichia coli P10486': {
                        'organism': 'Escherichia coli',
                        'uniquename': 'P10486',
                    }
                }
            },
        }
    }
    actual = canto_json
    add_gene_objects(actual, phi_df)
    assert actual == expected


def test_add_allele_objects():
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_allele_id': 'P10486:3a7f9b2e8c4d71e5-1',
                'allele_type': 'deletion',
                'canto_pathogen_gene_id': 'Escherichia coli P10486',
                'description': np.nan,
                'gene': 'hsdR',
                'name': 'hsdRdelta',
                'pathogen_gene_synonym': 'T1R1delta',
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_allele_id': 'P10486:3a7f9b2e8c4d71e5-2',
                'allele_type': 'wild_type',
                'canto_pathogen_gene_id': 'Escherichia coli P10486',
                'description': 'test',
                'gene': 'hsdR',
                'name': 'hsdR+',
                'pathogen_gene_synonym': np.nan,
            },
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'pathogen_allele_id': 'P10486:b6e0a5f3d92c8b17-1',
                'allele_type': 'deletion',
                'canto_pathogen_gene_id': 'Escherichia coli P10486',
                'description': np.nan,
                'gene': 'hsdR',
                'name': 'hsdRdelta',
                'pathogen_gene_synonym': ['FOO', 'BAR'],
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'alleles': {
                    'P10486:3a7f9b2e8c4d71e5-1': {
                        'allele_type': 'deletion',
                        'gene': 'Escherichia coli P10486',
                        'name': 'hsdRdelta',
                        'primary_identifier': 'P10486:3a7f9b2e8c4d71e5-1',
                        'synonyms': ['T1R1delta'],
                    },
                    'P10486:3a7f9b2e8c4d71e5-2': {
                        'allele_type': 'wild_type',
                        'description': 'test',
                        'gene': 'Escherichia coli P10486',
                        'name': 'hsdR+',
                        'primary_identifier': 'P10486:3a7f9b2e8c4d71e5-2',
                        'synonyms': [],
                    },
                }
            },
            'b6e0a5f3d92c8b17': {
                'alleles': {
                    'P10486:b6e0a5f3d92c8b17-1': {
                        'allele_type': 'deletion',
                        'gene': 'Escherichia coli P10486',
                        'name': 'hsdRdelta',
                        'primary_identifier': 'P10486:b6e0a5f3d92c8b17-1',
                        'synonyms': ['FOO', 'BAR'],
                    }
                }
            },
        }
    }
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'alleles': {}},
            'b6e0a5f3d92c8b17': {'alleles': {}},
        }
    }
    actual = canto_json
    add_allele_objects(actual, phi_df)
    assert actual == expected


def test_get_curation_date_df():
    phi_df = pd.DataFrame(
        {
            'session_id': ['3a7f9b2e8c4d71e5'],
            'curation_date': ['2023-01-12'],
        }
    )
    expected = pd.DataFrame.from_dict(
        {
            '3a7f9b2e8c4d71e5': {
                'start': '2023-01-12',
                'end': '2023-01-12',
                'created': '2023-01-12 09:00:00',
                'accepted': '2023-01-12 10:00:00',
                'curation_accepted': '2023-01-12 10:00:00',
                'curation_in_progress': '2023-01-12 11:00:00',
                'first_submitted': '2023-01-12 12:00:00',
                'first_approved': '2023-01-12 14:00:00',
                'needs_approval': '2023-01-12 12:00:00',
                'approval_in_progress': '2023-01-12 13:00:00',
                'approved': '2023-01-12 14:00:00',
                'annotation_status': '2023-01-12 14:00:00',
            }
        },
        orient='index',
    ).rename_axis('session_id')
    actual = get_curation_date_df(phi_df)
    pd.testing.assert_frame_equal(actual, expected)


def test_add_wild_type_genotype_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'genotypes': {}},
            'b6e0a5f3d92c8b17': {'genotypes': {}},
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'canto_host_genotype_id': 'Mus-musculus-wild-type-genotype-BALB/c',
                'host_strain': 'BALB/c',
                'host_id': '10090',
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'canto_host_genotype_id': 'Mus-musculus-wild-type-genotype-BALB/c',
                'host_strain': 'BALB/c',
                'host_id': '10090',
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'canto_host_genotype_id': 'Citrullus-lanatus-wild-type-genotype-cv.-Ruixin',
                'host_strain': 'cv. Ruixin',
                'host_id': '3654',
            },
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'canto_host_genotype_id': 'Mus-musculus-wild-type-genotype-BALB/c',
                'host_strain': 'BALB/c',
                'host_id': '10090',
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'genotypes': {
                    'Citrullus-lanatus-wild-type-genotype-cv.-Ruixin': {
                        'loci': [],
                        'organism_strain': 'cv. Ruixin',
                        'organism_taxonid': 3654,
                    },
                    'Mus-musculus-wild-type-genotype-BALB/c': {
                        'loci': [],
                        'organism_strain': 'BALB/c',
                        'organism_taxonid': 10090,
                    },
                },
            },
            'b6e0a5f3d92c8b17': {
                'genotypes': {
                    'Mus-musculus-wild-type-genotype-BALB/c': {
                        'loci': [],
                        'organism_strain': 'BALB/c',
                        'organism_taxonid': 10090,
                    }
                }
            },
        }
    }
    actual = canto_json
    add_wild_type_genotype_objects(actual, phi_df)
    assert actual == expected


def test_add_mutant_genotype_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'genotypes': {}},
            'b6e0a5f3d92c8b17': {'genotypes': {}},
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            # genotype with three alleles
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_id': 562,
                'pathogen_strain': 'K12',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-1',
                'pathogen_allele_id': 'P0A8V2:3a7f9b2e8c4d71e5-1',
                'expression': 'Overexpression',
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_id': 562,
                'pathogen_strain': 'K12',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-1',
                'pathogen_allele_id': 'P00579:3a7f9b2e8c4d71e5-1',
                'expression': 'Overexpression',
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_id': 562,
                'pathogen_strain': 'K12',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-1',
                'pathogen_allele_id': 'P08956:3a7f9b2e8c4d71e5-1',
                'expression': 'Not assayed',
            },
            # genotype from another species with one allele
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_id': 5518,
                'pathogen_strain': 'PH-1',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-2',
                'pathogen_allele_id': 'I1RF61:3a7f9b2e8c4d71e5-1',
                'expression': 'Knockdown',
            },
            # genotype from another session
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'pathogen_id': 562,
                'pathogen_strain': 'K12',
                'pathogen_genotype_id': 'b6e0a5f3d92c8b17-genotype-1',
                'pathogen_allele_id': 'P0A8V2:b6e0a5f3d92c8b17-1',
                'expression': 'Not assayed',
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'genotypes': {
                    '3a7f9b2e8c4d71e5-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Overexpression',
                                    'id': 'P00579:3a7f9b2e8c4d71e5-1',
                                }
                            ],
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P08956:3a7f9b2e8c4d71e5-1',
                                }
                            ],
                            [
                                {
                                    'expression': 'Overexpression',
                                    'id': 'P0A8V2:3a7f9b2e8c4d71e5-1',
                                }
                            ],
                        ],
                        'organism_strain': 'K12',
                        'organism_taxonid': 562,
                    },
                    '3a7f9b2e8c4d71e5-genotype-2': {
                        'loci': [
                            [
                                {
                                    'expression': 'Knockdown',
                                    'id': 'I1RF61:3a7f9b2e8c4d71e5-1',
                                }
                            ]
                        ],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                }
            },
            'b6e0a5f3d92c8b17': {
                'genotypes': {
                    'b6e0a5f3d92c8b17-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P0A8V2:b6e0a5f3d92c8b17-1',
                                }
                            ]
                        ],
                        'organism_strain': 'K12',
                        'organism_taxonid': 562,
                    }
                }
            },
        }
    }
    actual = canto_json
    add_mutant_genotype_objects(actual, phi_df)
    assert actual == expected


def test_add_metagenotype_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'metagenotypes': {}},
            'b6e0a5f3d92c8b17': {'metagenotypes': {}},
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'metagenotype_id': '3a7f9b2e8c4d71e5-metagenotype-1',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-1',
                'canto_host_genotype_id': 'Zea-mays-wild-type-genotype-B73-line',
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'metagenotype_id': '3a7f9b2e8c4d71e5-metagenotype-1',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-1',
                'canto_host_genotype_id': 'Zea-mays-wild-type-genotype-B73-line',
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'metagenotype_id': '3a7f9b2e8c4d71e5-metagenotype-2',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-2',
                'canto_host_genotype_id': 'Glycine-max-wild-type-genotype-Unknown-strain',
            },
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'metagenotype_id': 'b6e0a5f3d92c8b17-metagenotype-1',
                'pathogen_genotype_id': 'b6e0a5f3d92c8b17-genotype-1',
                'canto_host_genotype_id': 'Mus-musculus-wild-type-genotype-Unknown-strain',
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'metagenotypes': {
                    '3a7f9b2e8c4d71e5-metagenotype-1': {
                        'pathogen_genotype': '3a7f9b2e8c4d71e5-genotype-1',
                        'host_genotype': 'Zea-mays-wild-type-genotype-B73-line',
                        'type': 'pathogen-host',
                    },
                    '3a7f9b2e8c4d71e5-metagenotype-2': {
                        'pathogen_genotype': '3a7f9b2e8c4d71e5-genotype-2',
                        'host_genotype': 'Glycine-max-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                }
            },
            'b6e0a5f3d92c8b17': {
                'metagenotypes': {
                    'b6e0a5f3d92c8b17-metagenotype-1': {
                        'pathogen_genotype': 'b6e0a5f3d92c8b17-genotype-1',
                        'host_genotype': 'Mus-musculus-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    }
                }
            },
        }
    }
    actual = canto_json
    add_metagenotype_objects(actual, phi_df)
    assert actual == expected


def test_add_organism_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'organisms': {}},
            'b6e0a5f3d92c8b17': {'organisms': {}},
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_id': '562',
                'pathogen_species': 'Escherichia coli',
                'host_id': '10090',
                'host_species': 'Mus musculus',
            },
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'pathogen_id': '562',
                'pathogen_species': 'Escherichia coli',
                'host_id': '10090',
                'host_species': 'Mus musculus',
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'organisms': {
                    '562': {
                        'full_name': 'Escherichia coli',
                    },
                    '10090': {
                        'full_name': 'Mus musculus',
                    },
                }
            },
            'b6e0a5f3d92c8b17': {
                'organisms': {
                    '562': {
                        'full_name': 'Escherichia coli',
                    },
                    '10090': {
                        'full_name': 'Mus musculus',
                    },
                }
            },
        }
    }
    actual = canto_json
    add_organism_objects(actual, phi_df)
    assert actual == expected


def test_add_publication_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'publications': {}},
            'b6e0a5f3d92c8b17': {'publications': {}},
        }
    }
    phi_df = pd.DataFrame(
        {
            'session_id': [
                '3a7f9b2e8c4d71e5',
                '3a7f9b2e8c4d71e5',
                'b6e0a5f3d92c8b17',
            ],
            'pmid': [
                123,
                123,
                456,
            ],
        }
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'publications': {'PMID:123': {}}},
            'b6e0a5f3d92c8b17': {'publications': {'PMID:456': {}}},
        }
    }
    actual = canto_json
    add_publication_objects(actual, phi_df)
    assert actual == expected


def test_add_metadata_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'metadata': {}},
            'b6e0a5f3d92c8b17': {'metadata': {}},
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'protein_id': 'Q000909',
                'curation_date': '2023-01-12',
                'pmid': 123,
            },
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'protein_id': 'M4C4U1',
                'curation_date': '2023-01-12',
                'pmid': 123,
            },
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'protein_id': 'Q000909',
                'curation_date': '2023-02-15',
                'pmid': 456,
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'metadata': {
                    'accepted_timestamp': '2023-01-12 10:00:00',
                    'annotation_mode': 'standard',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2023-01-12 14:00:00',
                    'approval_in_progress_timestamp': '2023-01-12 13:00:00',
                    'approved_timestamp': '2023-01-12 14:00:00',
                    'canto_session': '3a7f9b2e8c4d71e5',
                    'curation_accepted_date': '2023-01-12 10:00:00',
                    'curation_in_progress_timestamp': '2023-01-12 11:00:00',
                    'curation_pub_id': 'PMID:123',
                    'curator_role': 'Molecular Connections',
                    'first_approved_timestamp': '2023-01-12 14:00:00',
                    'has_community_curation': False,
                    'needs_approval_timestamp': '2023-01-12 12:00:00',
                    'session_created_timestamp': '2023-01-12 09:00:00',
                    'session_first_submitted_timestamp': '2023-01-12 12:00:00',
                    'session_genes_count': '2',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                }
            },
            'b6e0a5f3d92c8b17': {
                'metadata': {
                    'accepted_timestamp': '2023-02-15 10:00:00',
                    'annotation_mode': 'standard',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2023-02-15 14:00:00',
                    'approval_in_progress_timestamp': '2023-02-15 13:00:00',
                    'approved_timestamp': '2023-02-15 14:00:00',
                    'canto_session': 'b6e0a5f3d92c8b17',
                    'curation_accepted_date': '2023-02-15 10:00:00',
                    'curation_in_progress_timestamp': '2023-02-15 11:00:00',
                    'curation_pub_id': 'PMID:456',
                    'curator_role': 'Molecular Connections',
                    'first_approved_timestamp': '2023-02-15 14:00:00',
                    'has_community_curation': False,
                    'needs_approval_timestamp': '2023-02-15 12:00:00',
                    'session_created_timestamp': '2023-02-15 09:00:00',
                    'session_first_submitted_timestamp': '2023-02-15 12:00:00',
                    'session_genes_count': '1',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                }
            },
        }
    }
    actual = canto_json
    add_metadata_objects(actual, phi_df)
    assert actual == expected


def test_add_disease_annotations():
    canto_json = {'curation_sessions': {'addf90acaa7ff81f': {'annotations': []}}}
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': 'addf90acaa7ff81f',
                'wt_metagenotype_id': 'addf90acaa7ff81f-metagenotype-20',
                'disease_id': 'PHIDO:0000120',
                'tissue': 'kidney',
                'tissue_id': 'BTO:0000671',
                'phi_ids': 'PHI:10643; PHI:10644; PHI:10645',
                'curation_date': '2020-09-01 00:00:00',
                'pmid': 32612591,
            }
        ]
    )
    phi_df.curation_date = pd.to_datetime(phi_df.curation_date)
    expected = {
        'curation_sessions': {
            'addf90acaa7ff81f': {
                'annotations': [
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2020-09-01',
                        'curator': {'community_curated': False},
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'kidney',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000671',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'figure': '',
                        'metagenotype': 'addf90acaa7ff81f-metagenotype-20',
                        'phi4_id': ['PHI:10643', 'PHI:10644', 'PHI:10645'],
                        'publication': 'PMID:32612591',
                        'status': 'new',
                        'term': 'PHIDO:0000120',
                        'type': 'disease_name',
                    }
                ]
            }
        }
    }
    actual = canto_json
    add_disease_annotations(actual, phi_df)
    assert actual == expected


def test_add_phenotype_annotations():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'annotations': []},
            'b6e0a5f3d92c8b17': {'annotations': []},
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'record_id': 'Record 1',
                'curation_date': pd.to_datetime('2023-01-12'),
                'phi_ids': 'PHI:1; PHI:2',
                'pmid': 123456,
                'comments': 'test comment',
                'tissue': 'leaf',
                'tissue_id': 'BTO:0000713',
                'metagenotype_id': '3a7f9b2e8c4d71e5-metagenotype-1',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-1',
                'canto_pathogen_gene_id': 'Fusarium graminearum Q00909',
            },
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'record_id': 'Record 2',
                'curation_date': pd.to_datetime('2023-02-12'),
                'phi_ids': 'PHI:7',
                'pmid': 458412,
                'comments': np.nan,
                'tissue': np.nan,
                'tissue_id': np.nan,
                'metagenotype_id': 'b6e0a5f3d92c8b17-metagenotype-1',
                'pathogen_genotype_id': 'b6e0a5f3d92c8b17-genotype-1',
                'canto_pathogen_gene_id': 'Hyaloperonospora arabidopsidis M4C4U1',
            },
        ]
    )
    phenotype_lookup = {
        'Record 1': [
            {
                'conditions': ['PECO:0005224'],
                'term': 'PHIPO:0000001',
                'extension': [
                    {
                        'rangeDisplayName': 'reduced virulence',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000015',
                        'relation': 'infective_ability',
                    },
                ],
                'feature_type': 'metagenotype',
            },
            {
                'conditions': ['PECO:0005224'],
                'term': 'PHIPO:0001210',
                # TODO: Find a pathogen phenotype extension
                'extension': [],
                'feature_type': 'pathogen_genotype',
            },
            {
                'conditions': [],
                'term': 'GO:0140418',
                # TODO: Find a GO annotation extension
                'extension': [],
                'feature_type': 'gene',
            },
        ],
        'Record 2': [
            {
                'conditions': [],
                'term': 'PHIPO:0000001',
                'extension': [],
                'feature_type': 'metagenotype',
            },
        ],
    }
    annotations_1 = [
        # metagenotype annotation
        {
            'checked': 'yes',
            'conditions': ['PECO:0005224'],
            'creation_date': '2023-01-12',
            'evidence_code': '',
            'extension': [
                {
                    'rangeDisplayName': 'reduced virulence',
                    'rangeType': 'Ontology',
                    'rangeValue': 'PHIPO:0000015',
                    'relation': 'infective_ability',
                },
                {
                    'rangeDisplayName': 'leaf',
                    'rangeType': 'Ontology',
                    'rangeValue': 'BTO:0000713',
                    'relation': 'infects_tissue',
                },
            ],
            'curator': {'community_curated': False},
            'figure': '',
            'phi4_id': ['PHI:1', 'PHI:2'],
            'publication': 'PMID:123456',
            'status': 'new',
            'term': 'PHIPO:0000001',
            'type': 'pathogen_host_interaction_phenotype',
            'submitter_comment': 'test comment',
            'metagenotype': '3a7f9b2e8c4d71e5-metagenotype-1',
        },
        # pathogen genotype annotation
        {
            'checked': 'yes',
            'conditions': ['PECO:0005224'],
            'creation_date': '2023-01-12',
            'evidence_code': '',
            'extension': [],
            'curator': {'community_curated': False},
            'figure': '',
            'phi4_id': ['PHI:1', 'PHI:2'],
            'publication': 'PMID:123456',
            'status': 'new',
            'term': 'PHIPO:0001210',
            'type': 'pathogen_phenotype',
            'submitter_comment': 'test comment',
            'genotype': '3a7f9b2e8c4d71e5-genotype-1',
        },
        # gene annotation
        {
            'checked': 'yes',
            'creation_date': '2023-01-12',
            'evidence_code': '',
            'extension': [],
            'curator': {'community_curated': False},
            'figure': '',
            'phi4_id': ['PHI:1', 'PHI:2'],
            'publication': 'PMID:123456',
            'status': 'new',
            'term': 'GO:0140418',
            'type': 'biological_process',
            'submitter_comment': 'test comment',
            'gene': 'Fusarium graminearum Q00909',
        },
    ]
    annotations_2 = [
        {
            'checked': 'yes',
            'conditions': [],
            'creation_date': '2023-02-12',
            'evidence_code': '',
            'extension': [],
            'curator': {'community_curated': False},
            'figure': '',
            'phi4_id': ['PHI:7'],
            'publication': 'PMID:458412',
            'status': 'new',
            'term': 'PHIPO:0000001',
            'type': 'pathogen_host_interaction_phenotype',
            'metagenotype': 'b6e0a5f3d92c8b17-metagenotype-1',
        },
    ]
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'annotations': annotations_1,
            },
            'b6e0a5f3d92c8b17': {
                'annotations': annotations_2,
            },
        }
    }
    actual = canto_json
    add_phenotype_annotations(actual, phenotype_lookup, phi_df)
    assert actual == expected


def test_add_wt_features():
    canto_json = {
        'curation_sessions': {
            'f86e79b5ed64a342': {
                'alleles': {
                    'P26215:f86e79b5ed64a342-1': {
                        'allele_type': 'disruption',
                        'gene': 'Bipolaris zeicola P26215',
                        'name': 'PGN1',
                        'primary_identifier': 'P26215:f86e79b5ed64a342-1',
                        'synonyms': [],
                        'gene_name': 'PGN1',
                    }
                },
                'genotypes': {
                    'f86e79b5ed64a342-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P26215:f86e79b5ed64a342-1',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                },
                'metagenotypes': {
                    'f86e79b5ed64a342-metagenotype-1': {
                        'pathogen_genotype': 'f86e79b5ed64a342-genotype-1',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    }
                },
            },
            '2d4bcfa71ccca168': {
                'alleles': {
                    'P22287:2d4bcfa71ccca168-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fulvia fulva P22287',
                        'name': 'AVR9delta',
                        'primary_identifier': 'P22287:2d4bcfa71ccca168-1',
                        'synonyms': [],
                        'gene_name': 'AVR9',
                    }
                },
                'genotypes': {
                    '2d4bcfa71ccca168-genotype-1': {
                        'loci': [
                            [
                                {
                                    'id': 'P22287:2d4bcfa71ccca168-1',
                                }
                            ]
                        ],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 5499,
                    },
                },
                'metagenotypes': {
                    '2d4bcfa71ccca168-metagenotype-1': {
                        'pathogen_genotype': '2d4bcfa71ccca168-genotype-1',
                        'host_genotype': 'Solanum-lycopersicum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    }
                },
            },
        }
    }
    all_feature_mapping = {
        'f86e79b5ed64a342': {
            'alleles': {
                'P26215:f86e79b5ed64a342-1': 'P26215:f86e79b5ed64a342-2',
            },
            'genotypes': {
                'f86e79b5ed64a342-genotype-1': 'f86e79b5ed64a342-genotype-2',
            },
            'metagenotypes': {
                'f86e79b5ed64a342-metagenotype-1': 'f86e79b5ed64a342-metagenotype-2',
            },
        },
        '2d4bcfa71ccca168': {
            'alleles': {
                'P22287:2d4bcfa71ccca168-1': 'P22287:2d4bcfa71ccca168-2',
            },
            'genotypes': {
                '2d4bcfa71ccca168-genotype-1': '2d4bcfa71ccca168-genotype-2',
            },
            'metagenotypes': {
                '2d4bcfa71ccca168-metagenotype-1': '2d4bcfa71ccca168-metagenotype-2',
            },
        },
    }
    expected = {
        'curation_sessions': {
            'f86e79b5ed64a342': {
                'alleles': {
                    'P26215:f86e79b5ed64a342-1': {
                        'allele_type': 'disruption',
                        'gene': 'Bipolaris zeicola P26215',
                        'name': 'PGN1',
                        'primary_identifier': 'P26215:f86e79b5ed64a342-1',
                        'synonyms': [],
                        'gene_name': 'PGN1',
                    },
                    'P26215:f86e79b5ed64a342-2': {
                        'allele_type': 'wild type',
                        'gene': 'Bipolaris zeicola P26215',
                        'name': 'PGN1+',
                        'primary_identifier': 'P26215:f86e79b5ed64a342-2',
                        'synonyms': [],
                        'gene_name': 'PGN1',
                    },
                },
                'genotypes': {
                    'f86e79b5ed64a342-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P26215:f86e79b5ed64a342-1',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                    'f86e79b5ed64a342-genotype-2': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P26215:f86e79b5ed64a342-2',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                },
                'metagenotypes': {
                    'f86e79b5ed64a342-metagenotype-1': {
                        'pathogen_genotype': 'f86e79b5ed64a342-genotype-1',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    'f86e79b5ed64a342-metagenotype-2': {
                        'pathogen_genotype': 'f86e79b5ed64a342-genotype-2',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
            },
            '2d4bcfa71ccca168': {
                'alleles': {
                    'P22287:2d4bcfa71ccca168-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fulvia fulva P22287',
                        'name': 'AVR9delta',
                        'primary_identifier': 'P22287:2d4bcfa71ccca168-1',
                        'synonyms': [],
                        'gene_name': 'AVR9',
                    },
                    'P22287:2d4bcfa71ccca168-2': {
                        'allele_type': 'wild type',
                        'gene': 'Fulvia fulva P22287',
                        'name': 'AVR9+',
                        'primary_identifier': 'P22287:2d4bcfa71ccca168-2',
                        'synonyms': [],
                        'gene_name': 'AVR9',
                    },
                },
                'genotypes': {
                    '2d4bcfa71ccca168-genotype-1': {
                        'loci': [
                            [
                                {
                                    'id': 'P22287:2d4bcfa71ccca168-1',
                                }
                            ]
                        ],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 5499,
                    },
                    '2d4bcfa71ccca168-genotype-2': {
                        'loci': [
                            [
                                {
                                    'id': 'P22287:2d4bcfa71ccca168-2',
                                    'expression': 'Not assayed',
                                }
                            ]
                        ],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 5499,
                    },
                },
                'metagenotypes': {
                    '2d4bcfa71ccca168-metagenotype-1': {
                        'pathogen_genotype': '2d4bcfa71ccca168-genotype-1',
                        'host_genotype': 'Solanum-lycopersicum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '2d4bcfa71ccca168-metagenotype-2': {
                        'pathogen_genotype': '2d4bcfa71ccca168-genotype-2',
                        'host_genotype': 'Solanum-lycopersicum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
            },
        }
    }
    actual = canto_json
    add_wt_features(all_feature_mapping, actual)
    assert actual == expected


def test_add_organism_roles():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'organisms': {
                    '0': {
                        'full_name': 'Imaginary species',
                    },
                    '161934': {
                        'full_name': 'Beta vulgaris',
                    },
                    '562': {
                        'full_name': 'Escherichia coli',
                    },
                }
            },
            # Empty on purpose, to test empty sessions
            'b6e0a5f3d92c8b17': {},
        }
    }
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'organisms': {
                    '0': {
                        'full_name': 'Imaginary species',
                        'role': 'unknown',
                    },
                    '161934': {
                        'full_name': 'Beta vulgaris',
                        'role': 'host',
                    },
                    '562': {
                        'full_name': 'Escherichia coli',
                        'role': 'pathogen',
                    },
                }
            },
            'b6e0a5f3d92c8b17': {},
        }
    }
    actual = canto_json
    add_organism_roles(actual)
    assert actual == expected


def test_add_pathogen_gene_names_to_alleles():
    canto_json = {
        'curation_sessions': {
            'b6e0a5f3d92c8b17': {
                'genes': {
                    'Fusarium graminearum Q00909': {
                        'uniquename': 'Q00909',
                    },
                },
                'alleles': {
                    'Q00909:b6e0a5f3d92c8b17-1': {
                        'gene': 'Fusarium graminearum Q00909',
                    }
                },
            }
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'protein_id': 'Q00909',
                'gene': 'TRI5',
            }
        ]
    )
    expected = {
        'curation_sessions': {
            'b6e0a5f3d92c8b17': {
                'genes': {
                    'Fusarium graminearum Q00909': {
                        'uniquename': 'Q00909',
                    },
                },
                'alleles': {
                    'Q00909:b6e0a5f3d92c8b17-1': {
                        'gene': 'Fusarium graminearum Q00909',
                        'gene_name': 'TRI5',
                    }
                },
            }
        }
    }
    actual = canto_json
    add_pathogen_gene_names_to_alleles(actual, phi_df)
    assert actual == expected


def test_get_phi_id_column():
    phi_df = pd.DataFrame.from_records(
        [
            {
                'phi_molconn_id': 'PHI:1',
                'multiple_mutation': np.nan,
            },
            {
                'phi_molconn_id': 'PHI:2',
                'multiple_mutation': 'PHI:11; PHI:10',
            },
            {
                'phi_molconn_id': 'PHI:3',
                'multiple_mutation': 'PHI:3; PHI:7; PHI:8',
            },
        ]
    )
    expected = pd.Series(
        [
            'PHI:1',
            'PHI:2; PHI:10; PHI:11',
            'PHI:3; PHI:7; PHI:8',
        ],
        name='phi_molconn_id',
    )
    actual = get_phi_id_column(phi_df)
    pd.testing.assert_series_equal(actual, expected)


def test_add_genotype_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'genotypes': {}},
            'b6e0a5f3d92c8b17': {'genotypes': {}},
        }
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'pathogen_id': 5518,
                'pathogen_strain': 'PH-1',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-2',
                'pathogen_allele_id': 'I1RF61:3a7f9b2e8c4d71e5-1',
                'expression': 'Knockdown',
                'canto_host_genotype_id': 'Citrullus-lanatus-wild-type-genotype-cv.-Ruixin',
                'host_strain': 'cv. Ruixin',
                'host_id': '3654',
            },
            {
                'session_id': 'b6e0a5f3d92c8b17',
                'pathogen_id': 5518,
                'pathogen_strain': 'PH-1',
                'pathogen_genotype_id': 'b6e0a5f3d92c8b17-genotype-2',
                'pathogen_allele_id': 'I1RF61:b6e0a5f3d92c8b17-1',
                'expression': 'Knockdown',
                'canto_host_genotype_id': 'Citrullus-lanatus-wild-type-genotype-cv.-Ruixin',
                'host_strain': 'cv. Ruixin',
                'host_id': '3654',
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'genotypes': {
                    '3a7f9b2e8c4d71e5-genotype-2': {
                        'loci': [
                            [
                                {
                                    'expression': 'Knockdown',
                                    'id': 'I1RF61:3a7f9b2e8c4d71e5-1',
                                }
                            ]
                        ],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                    'Citrullus-lanatus-wild-type-genotype-cv.-Ruixin': {
                        'loci': [],
                        'organism_strain': 'cv. Ruixin',
                        'organism_taxonid': 3654,
                    },
                },
            },
            'b6e0a5f3d92c8b17': {
                'genotypes': {
                    'b6e0a5f3d92c8b17-genotype-2': {
                        'loci': [
                            [
                                {
                                    'expression': 'Knockdown',
                                    'id': 'I1RF61:b6e0a5f3d92c8b17-1',
                                }
                            ]
                        ],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                    'Citrullus-lanatus-wild-type-genotype-cv.-Ruixin': {
                        'loci': [],
                        'organism_strain': 'cv. Ruixin',
                        'organism_taxonid': 3654,
                    },
                },
            },
        }
    }
    actual = canto_json
    add_genotype_objects(canto_json, phi_df)
    assert actual == expected


def test_make_phenotype_mapping():
    phenotype_mapping_df = pd.DataFrame.from_records(
        [
            {
                'column_1': 'mutant_phenotype',
                'value_1': 'unaffected pathogenicity',
                'column_2': np.nan,
                'value_2': np.nan,
                'primary_id': np.nan,
                'primary_label': np.nan,
                'eco_id': np.nan,
                'eco_label': np.nan,
                'feature': 'metagenotype',
                'extension_relation': 'infective_ability',
                'extension_range': 'PHIPO:0000004',
            },
            {
                'column_1': 'in_vitro_growth',
                'value_1': 'defective on minimal media',
                'column_2': 'is_filamentous',
                'value_2': False,
                'primary_id': 'PHIPO:0000974',
                'primary_label': 'decreased unicellular population growth',
                'eco_id': 'PECO:0005020',
                'eco_label': 'minimal medium',
                'feature': 'pathogen_genotype',
                'extension_relation': np.nan,
                'extension_range': np.nan,
            },
            {
                'column_1': 'mutant_phenotype',
                'value_1': 'effector (plant avirulence determinant)',
                'column_2': np.nan,
                'value_2': np.nan,
                'primary_id': 'GO:0140418',
                'primary_label': 'effector-mediated modulation of host process by symbiont',
                'eco_id': np.nan,
                'eco_label': np.nan,
                'feature': 'gene',
                'extension_relation': np.nan,
                'extension_range': np.nan,
            },
            {
                'column_1': 'postpenetration_defect',
                'value_1': 'no',
                'column_2': np.nan,
                'value_2': np.nan,
                'primary_id': 'PHIPO:0000954',
                'primary_label': 'presence of pathogen growth within host',
                'eco_id': np.nan,
                'eco_label': np.nan,
                'feature': 'metagenotype',
                'extension_relation': np.nan,
                'extension_range': np.nan,
            },
        ],
    )
    phipo_mapping = {
        'PHIPO:0000004': 'unaffected pathogenicity',
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'record_id': 'Record 1',
                'mutant_phenotype': 'unaffected pathogenicity',
                'in_vitro_growth': 'defective on minimal media',
                'is_filamentous': False,
                'postpenetration_defect': 'no',
            },
            {
                'record_id': 'Record 2',
                'mutant_phenotype': 'effector (plant avirulence determinant)',
                'in_vitro_growth': np.nan,
                'is_filamentous': np.nan,
                'postpenetration_defect': np.nan,
            },
        ]
    )
    expected = {
        'Record 1': [
            {
                'conditions': ['PECO:0005020'],
                'term': 'PHIPO:0000974',
                'extension': [],
                'feature_type': 'pathogen_genotype',
            },
            {
                'conditions': [],
                'term': 'PHIPO:0000001',
                'extension': [
                    {
                        'rangeDisplayName': 'unaffected pathogenicity',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000004',
                        'relation': 'infective_ability',
                    }
                ],
                'feature_type': 'metagenotype',
            },
            {
                'conditions': [],
                'term': 'PHIPO:0000954',
                'extension': [],
                'feature_type': 'metagenotype',
            },
        ],
        'Record 2': [
            {
                'conditions': [],
                'term': 'GO:0140418',
                'extension': [],
                'feature_type': 'gene',
            }
        ],
    }
    actual = make_phenotype_mapping(phenotype_mapping_df, phipo_mapping, phi_df)
    assert actual == expected


def test_add_annotation_objects():
    canto_json = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {'annotations': []},
        }
    }
    phenotype_lookup = phenotype_lookup = {
        'Record 1': [
            {
                'conditions': ['PECO:0005224'],
                'term': 'PHIPO:0000001',
                'extension': [
                    {
                        'rangeDisplayName': 'reduced virulence',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000015',
                        'relation': 'infective_ability',
                    },
                ],
                'feature_type': 'metagenotype',
            },
        ]
    }
    phi_df = pd.DataFrame.from_records(
        [
            {
                'session_id': '3a7f9b2e8c4d71e5',
                'record_id': 'Record 1',
                'curation_date': pd.to_datetime('2023-01-12'),
                'phi_ids': 'PHI:1; PHI:2',
                'pmid': 123456,
                'comments': 'test comment',
                'tissue': 'leaf',
                'tissue_id': 'BTO:0000713',
                'metagenotype_id': '3a7f9b2e8c4d71e5-metagenotype-1',
                'wt_metagenotype_id': '3a7f9b2e8c4d71e5-metagenotype-2',
                'disease_id': 'PHIDO:0000120',
                'pathogen_genotype_id': '3a7f9b2e8c4d71e5-genotype-1',
                'canto_pathogen_gene_id': 'Fusarium graminearum Q00909',
            },
        ]
    )
    expected = {
        'curation_sessions': {
            '3a7f9b2e8c4d71e5': {
                'annotations': [
                    {
                        'checked': 'yes',
                        'conditions': ['PECO:0005224'],
                        'creation_date': '2023-01-12',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'reduced virulence',
                                'rangeType': 'Ontology',
                                'rangeValue': 'PHIPO:0000015',
                                'relation': 'infective_ability',
                            },
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            },
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:1', 'PHI:2'],
                        'publication': 'PMID:123456',
                        'status': 'new',
                        'term': 'PHIPO:0000001',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'test comment',
                        'metagenotype': '3a7f9b2e8c4d71e5-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2023-01-12',
                        'curator': {'community_curated': False},
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'figure': '',
                        'metagenotype': '3a7f9b2e8c4d71e5-metagenotype-2',
                        'phi4_id': ['PHI:1', 'PHI:2'],
                        'publication': 'PMID:123456',
                        'status': 'new',
                        'term': 'PHIDO:0000120',
                        'type': 'disease_name',
                    },
                ]
            }
        }
    }
    actual = canto_json
    add_annotation_objects(canto_json, phenotype_lookup, phi_df)
    assert actual == expected


def test_make_phibase_json():
    phibase_path = DATA_DIR / 'phibase_test.csv'
    approved_pmids = [15894715]
    expected = {
        'curation_sessions': {
            'f86e79b5ed64a342': {
                'alleles': {
                    'P26215:f86e79b5ed64a342-1': {
                        'allele_type': 'disruption',
                        'gene': 'Bipolaris zeicola P26215',
                        'name': 'PGN1',
                        'primary_identifier': 'P26215:f86e79b5ed64a342-1',
                        'synonyms': [],
                        'gene_name': 'PGN1',
                    },
                    'P26215:f86e79b5ed64a342-2': {
                        'allele_type': 'wild type',
                        'gene': 'Bipolaris zeicola P26215',
                        'name': 'PGN1+',
                        'primary_identifier': 'P26215:f86e79b5ed64a342-2',
                        'synonyms': [],
                        'gene_name': 'PGN1',
                    },
                },
                'annotations': [
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'unaffected pathogenicity',
                                'rangeType': 'Ontology',
                                'rangeValue': 'PHIPO:0000004',
                                'relation': 'infective_ability',
                            },
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            },
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000001',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000331',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000954',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000024',
                        'type': 'pathogen_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'genotype': 'f86e79b5ed64a342-genotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'curator': {'community_curated': False},
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'figure': '',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-2',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIDO:0000211',
                        'type': 'disease_name',
                    },
                ],
                'genes': {
                    'Bipolaris zeicola P26215': {
                        'organism': 'Bipolaris zeicola',
                        'uniquename': 'P26215',
                    }
                },
                'genotypes': {
                    'f86e79b5ed64a342-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P26215:f86e79b5ed64a342-1',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                    'Zea-mays-wild-type-genotype-Unknown-strain': {
                        'loci': [],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 4577,
                    },
                    'f86e79b5ed64a342-genotype-2': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P26215:f86e79b5ed64a342-2',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                },
                'metadata': {
                    'accepted_timestamp': '2000-05-10 10:00:00',
                    'annotation_mode': 'standard',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2000-05-10 14:00:00',
                    'approval_in_progress_timestamp': '2000-05-10 13:00:00',
                    'approved_timestamp': '2000-05-10 14:00:00',
                    'canto_session': 'f86e79b5ed64a342',
                    'curation_accepted_date': '2000-05-10 10:00:00',
                    'curation_in_progress_timestamp': '2000-05-10 11:00:00',
                    'curation_pub_id': 'PMID:2152162',
                    'curator_role': 'Molecular Connections',
                    'first_approved_timestamp': '2000-05-10 14:00:00',
                    'has_community_curation': False,
                    'needs_approval_timestamp': '2000-05-10 12:00:00',
                    'session_created_timestamp': '2000-05-10 09:00:00',
                    'session_first_submitted_timestamp': '2000-05-10 12:00:00',
                    'session_genes_count': '1',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {
                    'f86e79b5ed64a342-metagenotype-1': {
                        'pathogen_genotype': 'f86e79b5ed64a342-genotype-1',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    'f86e79b5ed64a342-metagenotype-2': {
                        'pathogen_genotype': 'f86e79b5ed64a342-genotype-2',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
                'organisms': {
                    '5017': {'full_name': 'Bipolaris zeicola'},
                    '4577': {'full_name': 'Zea mays'},
                },
                'publications': {'PMID:2152162': {}},
            },
            '2d4bcfa71ccca168': {
                'alleles': {
                    'P22287:2d4bcfa71ccca168-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fulvia fulva P22287',
                        'name': 'AVR9delta',
                        'primary_identifier': 'P22287:2d4bcfa71ccca168-1',
                        'synonyms': [],
                        'gene_name': 'AVR9',
                    },
                    'P22287:2d4bcfa71ccca168-2': {
                        'allele_type': 'wild type',
                        'gene': 'Fulvia fulva P22287',
                        'name': 'AVR9+',
                        'primary_identifier': 'P22287:2d4bcfa71ccca168-2',
                        'synonyms': [],
                        'gene_name': 'AVR9',
                    },
                },
                'annotations': [
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:7'],
                        'publication': 'PMID:1799694',
                        'status': 'new',
                        'term': 'PHIPO:0000001',
                        'type': 'pathogen_host_interaction_phenotype',
                        'metagenotype': '2d4bcfa71ccca168-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:7'],
                        'publication': 'PMID:1799694',
                        'status': 'new',
                        'term': 'GO:0140418',
                        'type': 'biological_process',
                        'gene': 'Fulvia fulva P22287',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'curator': {'community_curated': False},
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'figure': '',
                        'metagenotype': '2d4bcfa71ccca168-metagenotype-2',
                        'phi4_id': ['PHI:7'],
                        'publication': 'PMID:1799694',
                        'status': 'new',
                        'term': 'PHIDO:0000209',
                        'type': 'disease_name',
                    },
                ],
                'genes': {
                    'Fulvia fulva P22287': {
                        'organism': 'Fulvia fulva',
                        'uniquename': 'P22287',
                    }
                },
                'genotypes': {
                    '2d4bcfa71ccca168-genotype-1': {
                        'loci': [[{'id': 'P22287:2d4bcfa71ccca168-1'}]],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 5499,
                    },
                    'Solanum-lycopersicum-wild-type-genotype-Unknown-strain': {
                        'loci': [],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 4081,
                    },
                    '2d4bcfa71ccca168-genotype-2': {
                        'loci': [
                            [
                                {
                                    'id': 'P22287:2d4bcfa71ccca168-2',
                                    'expression': 'Not assayed',
                                }
                            ]
                        ],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 5499,
                    },
                },
                'metadata': {
                    'accepted_timestamp': '2000-05-10 10:00:00',
                    'annotation_mode': 'standard',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2000-05-10 14:00:00',
                    'approval_in_progress_timestamp': '2000-05-10 13:00:00',
                    'approved_timestamp': '2000-05-10 14:00:00',
                    'canto_session': '2d4bcfa71ccca168',
                    'curation_accepted_date': '2000-05-10 10:00:00',
                    'curation_in_progress_timestamp': '2000-05-10 11:00:00',
                    'curation_pub_id': 'PMID:1799694',
                    'curator_role': 'Molecular Connections',
                    'first_approved_timestamp': '2000-05-10 14:00:00',
                    'has_community_curation': False,
                    'needs_approval_timestamp': '2000-05-10 12:00:00',
                    'session_created_timestamp': '2000-05-10 09:00:00',
                    'session_first_submitted_timestamp': '2000-05-10 12:00:00',
                    'session_genes_count': '1',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {
                    '2d4bcfa71ccca168-metagenotype-1': {
                        'pathogen_genotype': '2d4bcfa71ccca168-genotype-1',
                        'host_genotype': 'Solanum-lycopersicum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '2d4bcfa71ccca168-metagenotype-2': {
                        'pathogen_genotype': '2d4bcfa71ccca168-genotype-2',
                        'host_genotype': 'Solanum-lycopersicum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
                'organisms': {
                    '5499': {'full_name': 'Fulvia fulva'},
                    '4081': {'full_name': 'Solanum lycopersicum'},
                },
                'publications': {'PMID:1799694': {}},
            },
            '7ce9018575492179': {
                'alleles': {
                    'Q01886:7ce9018575492179-1': {
                        'allele_type': 'disruption',
                        'gene': 'Bipolaris zeicola Q01886',
                        'name': 'HTS1',
                        'primary_identifier': 'Q01886:7ce9018575492179-1',
                        'synonyms': [],
                        'gene_name': 'HTS1',
                    },
                    'Q01886:7ce9018575492179-2': {
                        'allele_type': 'wild type',
                        'gene': 'Bipolaris zeicola Q01886',
                        'name': 'HTS1+',
                        'primary_identifier': 'Q01886:7ce9018575492179-2',
                        'synonyms': [],
                        'gene_name': 'HTS1',
                    },
                },
                'annotations': [
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2005-05-04',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'loss of pathogenicity',
                                'rangeType': 'Ontology',
                                'rangeValue': 'PHIPO:0000010',
                                'relation': 'infective_ability',
                            },
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            },
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:12'],
                        'publication': 'PMID:11607305',
                        'status': 'new',
                        'term': 'PHIPO:0000001',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'metagenotype': '7ce9018575492179-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2005-05-04',
                        'curator': {'community_curated': False},
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'figure': '',
                        'metagenotype': '7ce9018575492179-metagenotype-2',
                        'phi4_id': ['PHI:12'],
                        'publication': 'PMID:11607305',
                        'status': 'new',
                        'term': 'PHIDO:0000211',
                        'type': 'disease_name',
                    },
                ],
                'genes': {
                    'Bipolaris zeicola Q01886': {
                        'organism': 'Bipolaris zeicola',
                        'uniquename': 'Q01886',
                    }
                },
                'genotypes': {
                    '7ce9018575492179-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'Q01886:7ce9018575492179-1',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                    'Zea-mays-wild-type-genotype-Unknown-strain': {
                        'loci': [],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 4577,
                    },
                    '7ce9018575492179-genotype-2': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'Q01886:7ce9018575492179-2',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                },
                'metadata': {
                    'accepted_timestamp': '2005-05-04 10:00:00',
                    'annotation_mode': 'standard',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2005-05-04 14:00:00',
                    'approval_in_progress_timestamp': '2005-05-04 13:00:00',
                    'approved_timestamp': '2005-05-04 14:00:00',
                    'canto_session': '7ce9018575492179',
                    'curation_accepted_date': '2005-05-04 10:00:00',
                    'curation_in_progress_timestamp': '2005-05-04 11:00:00',
                    'curation_pub_id': 'PMID:11607305',
                    'curator_role': 'Molecular Connections',
                    'first_approved_timestamp': '2005-05-04 14:00:00',
                    'has_community_curation': False,
                    'needs_approval_timestamp': '2005-05-04 12:00:00',
                    'session_created_timestamp': '2005-05-04 09:00:00',
                    'session_first_submitted_timestamp': '2005-05-04 12:00:00',
                    'session_genes_count': '1',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {
                    '7ce9018575492179-metagenotype-1': {
                        'pathogen_genotype': '7ce9018575492179-genotype-1',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '7ce9018575492179-metagenotype-2': {
                        'pathogen_genotype': '7ce9018575492179-genotype-2',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
                'organisms': {
                    '5017': {'full_name': 'Bipolaris zeicola'},
                    '4577': {'full_name': 'Zea mays'},
                },
                'publications': {'PMID:11607305': {}},
            },
        },
        'schema_version': 1,
    }
    actual = make_phibase_json(phibase_path, approved_pmids)
    assert actual == expected


def test_make_combined_export():
    phibase_path = DATA_DIR / 'combined_export_test.csv'
    phicanto_path = DATA_DIR / 'combined_export_test.json'
    expected = {
        'curation_sessions': {
            'f86e79b5ed64a342': {
                'alleles': {
                    'P26215:f86e79b5ed64a342-1': {
                        'allele_type': 'disruption',
                        'gene': 'Bipolaris zeicola P26215',
                        'name': 'PGN1',
                        'primary_identifier': 'P26215:f86e79b5ed64a342-1',
                        'synonyms': [],
                    },
                    'P26215:f86e79b5ed64a342-2': {
                        'allele_type': 'wild type',
                        'gene': 'Bipolaris zeicola P26215',
                        'name': 'PGN1+',
                        'primary_identifier': 'P26215:f86e79b5ed64a342-2',
                        'synonyms': [],
                    },
                },
                'annotations': [
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'unaffected pathogenicity',
                                'rangeType': 'Ontology',
                                'rangeValue': 'PHIPO:0000004',
                                'relation': 'infective_ability',
                            },
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            },
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000001',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000331',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000954',
                        'type': 'pathogen_host_interaction_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIPO:0000024',
                        'type': 'pathogen_phenotype',
                        'submitter_comment': 'Expression during all infection stages. pathogen formerly called Cochliobolus carbonum teleomorph name',
                        'genotype': 'f86e79b5ed64a342-genotype-1',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'curator': {'community_curated': False},
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'figure': '',
                        'metagenotype': 'f86e79b5ed64a342-metagenotype-2',
                        'phi4_id': ['PHI:3'],
                        'publication': 'PMID:2152162',
                        'status': 'new',
                        'term': 'PHIDO:0000211',
                        'type': 'disease_name',
                    },
                ],
                'genes': {
                    'Bipolaris zeicola P26215': {
                        'organism': 'Bipolaris zeicola',
                        'uniquename': 'P26215',
                    }
                },
                'genotypes': {
                    'f86e79b5ed64a342-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P26215:f86e79b5ed64a342-1',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                    'Zea-mays-wild-type-genotype-Unknown-strain': {
                        'loci': [],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 4577,
                    },
                    'f86e79b5ed64a342-genotype-2': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'P26215:f86e79b5ed64a342-2',
                                }
                            ]
                        ],
                        'organism_strain': 'SB111',
                        'organism_taxonid': 5017,
                    },
                },
                'metadata': {
                    'accepted_timestamp': '2000-05-10 10:00:00',
                    'annotation_mode': 'standard',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2000-05-10 14:00:00',
                    'approval_in_progress_timestamp': '2000-05-10 13:00:00',
                    'approved_timestamp': '2000-05-10 14:00:00',
                    'canto_session': 'f86e79b5ed64a342',
                    'curation_accepted_date': '2000-05-10 10:00:00',
                    'curation_in_progress_timestamp': '2000-05-10 11:00:00',
                    'curation_pub_id': 'PMID:2152162',
                    'curator_role': 'Molecular Connections',
                    'first_approved_timestamp': '2000-05-10 14:00:00',
                    'has_community_curation': False,
                    'needs_approval_timestamp': '2000-05-10 12:00:00',
                    'session_created_timestamp': '2000-05-10 09:00:00',
                    'session_first_submitted_timestamp': '2000-05-10 12:00:00',
                    'session_genes_count': '1',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {
                    'f86e79b5ed64a342-metagenotype-1': {
                        'pathogen_genotype': 'f86e79b5ed64a342-genotype-1',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    'f86e79b5ed64a342-metagenotype-2': {
                        'pathogen_genotype': 'f86e79b5ed64a342-genotype-2',
                        'host_genotype': 'Zea-mays-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
                'organisms': {
                    '5017': {'full_name': 'Bipolaris zeicola', 'role': 'pathogen'},
                    '4577': {'full_name': 'Zea mays', 'role': 'host'},
                },
                'publications': {'PMID:2152162': {}},
            },
            '2d4bcfa71ccca168': {
                'alleles': {
                    'P22287:2d4bcfa71ccca168-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fulvia fulva P22287',
                        'name': 'AVR9Δ',
                        'primary_identifier': 'P22287:2d4bcfa71ccca168-1',
                        'synonyms': [],
                    },
                    'P22287:2d4bcfa71ccca168-2': {
                        'allele_type': 'wild type',
                        'gene': 'Fulvia fulva P22287',
                        'name': 'AVR9+',
                        'primary_identifier': 'P22287:2d4bcfa71ccca168-2',
                        'synonyms': [],
                    },
                },
                'annotations': [
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:7'],
                        'publication': 'PMID:1799694',
                        'status': 'new',
                        'term': 'PHIPO:0000001',
                        'type': 'pathogen_host_interaction_phenotype',
                        'metagenotype': '2d4bcfa71ccca168-metagenotype-1',
                    },
                    {
                        'checked': 'yes',
                        'creation_date': '2000-05-10',
                        'evidence_code': '',
                        'extension': [],
                        'curator': {'community_curated': False},
                        'figure': '',
                        'phi4_id': ['PHI:7'],
                        'publication': 'PMID:1799694',
                        'status': 'new',
                        'term': 'GO:0140418',
                        'type': 'biological_process',
                        'gene': 'Fulvia fulva P22287',
                    },
                    {
                        'checked': 'yes',
                        'conditions': [],
                        'creation_date': '2000-05-10',
                        'curator': {'community_curated': False},
                        'evidence_code': '',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            }
                        ],
                        'figure': '',
                        'metagenotype': '2d4bcfa71ccca168-metagenotype-2',
                        'phi4_id': ['PHI:7'],
                        'publication': 'PMID:1799694',
                        'status': 'new',
                        'term': 'PHIDO:0000209',
                        'type': 'disease_name',
                    },
                ],
                'genes': {
                    'Fulvia fulva P22287': {
                        'organism': 'Fulvia fulva',
                        'uniquename': 'P22287',
                    }
                },
                'genotypes': {
                    '2d4bcfa71ccca168-genotype-1': {
                        'loci': [[{'id': 'P22287:2d4bcfa71ccca168-1'}]],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 5499,
                    },
                    'Solanum-lycopersicum-wild-type-genotype-Unknown-strain': {
                        'loci': [],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 4081,
                    },
                    '2d4bcfa71ccca168-genotype-2': {
                        'loci': [
                            [
                                {
                                    'id': 'P22287:2d4bcfa71ccca168-2',
                                    'expression': 'Not assayed',
                                }
                            ]
                        ],
                        'organism_strain': 'Unknown strain',
                        'organism_taxonid': 5499,
                    },
                },
                'metadata': {
                    'accepted_timestamp': '2000-05-10 10:00:00',
                    'annotation_mode': 'standard',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2000-05-10 14:00:00',
                    'approval_in_progress_timestamp': '2000-05-10 13:00:00',
                    'approved_timestamp': '2000-05-10 14:00:00',
                    'canto_session': '2d4bcfa71ccca168',
                    'curation_accepted_date': '2000-05-10 10:00:00',
                    'curation_in_progress_timestamp': '2000-05-10 11:00:00',
                    'curation_pub_id': 'PMID:1799694',
                    'curator_role': 'Molecular Connections',
                    'first_approved_timestamp': '2000-05-10 14:00:00',
                    'has_community_curation': False,
                    'needs_approval_timestamp': '2000-05-10 12:00:00',
                    'session_created_timestamp': '2000-05-10 09:00:00',
                    'session_first_submitted_timestamp': '2000-05-10 12:00:00',
                    'session_genes_count': '1',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {
                    '2d4bcfa71ccca168-metagenotype-1': {
                        'pathogen_genotype': '2d4bcfa71ccca168-genotype-1',
                        'host_genotype': 'Solanum-lycopersicum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '2d4bcfa71ccca168-metagenotype-2': {
                        'pathogen_genotype': '2d4bcfa71ccca168-genotype-2',
                        'host_genotype': 'Solanum-lycopersicum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
                'organisms': {
                    '5499': {'full_name': 'Fulvia fulva', 'role': 'pathogen'},
                    '4081': {'full_name': 'Solanum lycopersicum', 'role': 'host'},
                },
                'publications': {'PMID:1799694': {}},
            },
            '144c8f8625c95e5e': {
                'alleles': {
                    'Q7WT47:144c8f8625c95e5e-7': {
                        'allele_type': 'nucleotide_insertion',
                        'description': '618-A',
                        'gene': 'Enterococcus faecalis Q7WT47',
                        'name': 'lsa',
                        'primary_identifier': 'Q7WT47:144c8f8625c95e5e-7',
                        'synonyms': [],
                    }
                },
                'annotations': [
                    {
                        'conditions': [
                            'PECO:0000102',
                            'PECO:0005225',
                            'PECO:0000004',
                            'PECO:0005186',
                        ],
                        'creation_date': '2023-11-03',
                        'curator': {'community_curated': True},
                        'evidence_code': 'Cell growth assay',
                        'extension': [
                            {
                                'rangeDisplayName': 'high',
                                'rangeType': 'Ontology',
                                'rangeValue': 'FYPO_EXT:0000001',
                                'relation': 'has_severity',
                            }
                        ],
                        'figure': 'Table 1',
                        'genotype': '144c8f8625c95e5e-genotype-1',
                        'publication': 'PMID:12821484',
                        'status': 'new',
                        'submitter_comment': '',
                        'term': 'PHIPO:0000560',
                        'type': 'pathogen_phenotype',
                    }
                ],
                'genes': {
                    'Enterococcus faecalis Q7WT47': {
                        'organism': 'Enterococcus faecalis',
                        'uniquename': 'Q7WT47',
                    }
                },
                'genotypes': {
                    '144c8f8625c95e5e-genotype-1': {
                        'loci': [
                            [
                                {
                                    'expression': 'Not assayed',
                                    'id': 'Q7WT47:144c8f8625c95e5e-7',
                                }
                            ]
                        ],
                        'organism_strain': 'UCN32',
                        'organism_taxonid': 1351,
                    }
                },
                'metadata': {
                    'accepted_timestamp': '2023-11-02 23:41:46',
                    'annotation_mode': 'advanced',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2024-01-31 11:51:14',
                    'approval_in_progress_timestamp': '2024-01-31 11:51:12',
                    'approved_timestamp': '2024-01-31 11:51:14',
                    'canto_session': '144c8f8625c95e5e',
                    'curation_accepted_date': '2023-11-02 23:41:46',
                    'curation_in_progress_timestamp': '2023-11-02 23:41:56',
                    'curation_pub_id': 'PMID:12821484',
                    'curator_role': 'community',
                    'first_approved_timestamp': '2024-01-31 11:51:14',
                    'has_community_curation': True,
                    'needs_approval_timestamp': '2024-01-31 11:51:09',
                    'session_created_timestamp': '2023-11-02 23:41:39',
                    'session_first_submitted_timestamp': '2024-01-31 11:51:09',
                    'session_genes_count': '1',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'organisms': {
                    '1351': {'full_name': 'Enterococcus faecalis', 'role': 'pathogen'}
                },
                'publications': {'PMID:12821484': {}},
            },
        },
        'schema_version': 1,
    }
    actual = make_combined_export(phibase_path, phicanto_path)
    assert actual == expected
