# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import importlib.resources
from pathlib import Path

import numpy as np
import pytest
from phibase_pipeline import loaders

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal 


TEST_DATA_DIR = Path(__file__).parent / 'data'
ENSEMBL_DATA_DIR = TEST_DATA_DIR / 'ensembl'


@pytest.fixture
def chemical_data():
    return {
        'PHIPO:0000707': {
            'chebi_id': 'CHEBI:5827',
            'chebi_label': 'hymexazol',
            'frac': '32',
            'cas': '10004-44-1',
            'smiles': 'Cc1cc(O)no1',
        },
        'PHIPO:0000581': {
            'chebi_id': 'CHEBI:5100',
            'chebi_label': 'flucytosine',
            'frac': None,
            'cas': '2022-85-7',
            'smiles': 'Nc1nc(=O)[nH]cc1F',
        },
        'PHIPO:0000647': {
            'chebi_id': 'CHEBI:39214',
            'chebi_label': 'abamectin',
            'frac': None,
            'cas': '71751-41-2',
            'smiles': None,
        },
        'PHIPO:0000545': {
            'chebi_id': 'CHEBI:3639',
            'chebi_label': 'chlorothalonil',
            'frac': 'M05',
            'cas': None,
            'smiles': 'Clc1c(Cl)c(C#N)c(Cl)c(C#N)c1Cl',
        },
        'PHIPO:0000705': {
            'chebi_id': None,
            'chebi_label': 'norfloxacin',
            'frac': None,
            'cas': '70458-96-7',
            'smiles': None,
        },
        'PHIPO:0000690': {
            'chebi_id': 'CHEBI:28368',
            'chebi_label': None,
            'frac': None,
            'cas': '303-81-1',
            'smiles': None,
        },
    }


def test_read_phig_uniprot_mapping():
    path = TEST_DATA_DIR / 'phig_uniprot_mapping.csv'
    expected = {
        'A0A059ZHI3': 'PHIG:3546',
        'A0A059ZQI8': 'PHIG:1040',
        'A0A059ZR97': 'PHIG:3696',
        'I1RAE5': 'PHIG:11',
        'G4MXR2': 'PHIG:294',
    }
    actual = loaders.read_phig_uniprot_mapping(path)
    assert actual == expected


def test_read_phipo_chebi_mapping():
    path = ENSEMBL_DATA_DIR / 'phipo_chebi_mapping.csv'
    actual = loaders.read_phipo_chebi_mapping(path)
    expected = {
        'PHIPO:0000647': {'id': 'CHEBI:39214', 'label': 'abamectin'},
        'PHIPO:0000534': {'id': 'CHEBI:27666', 'label': 'actinomycin D'},
        'PHIPO:0000591': {'id': 'CHEBI:53661', 'label': 'alexidine'},
        'PHIPO:0000592': {'id': 'CHEBI:81763', 'label': 'fludioxonil'},
    }
    assert actual == expected


def test_load_in_vitro_growth_classifier():
    expected = pd.Series(
        data={
            28447: True,
            645: False,
            5518: True,
        },
        name='is_filamentous',
    ).rename_axis('ncbi_taxid')
    actual = loaders.load_in_vitro_growth_classifier(
        TEST_DATA_DIR / 'in_vitro_growth_mapping.csv'
    )
    pd.testing.assert_series_equal(actual, expected)


def test_load_disease_column_mapping():
    expected = {
        # From PHIDO
        'american foulbrood': 'PHIDO:0000011',
        'angular leaf spot': 'PHIDO:0000012',
        'anthracnose': 'PHIDO:0000013',
        'anthracnose leaf spot': 'PHIDO:0000014',
        'fusarium head blight': 'PHIDO:0000163',
        # From disease mapping
        'airsacculitis': 'PHIDO:0000008',
        'bacterial canker': 'PHIDO:0000025',
        'anthracnose (cruciferae)': 'PHIDO:0000013',
        'anthracnose (cucurbitaceae)': 'PHIDO:0000013',
        'anthracnose (cucumber)': 'PHIDO:0000013',
        'biocontrol: non pathogenic': np.nan,
        'blight': 'blight',
        'bacterial wilt; bacterial canker': 'PHIDO:0000035; PHIDO:0000025'
    }
    actual = loaders.load_disease_column_mapping(
        phido_path=TEST_DATA_DIR / 'phido.csv',
        extra_path=TEST_DATA_DIR / 'disease_mapping.csv',
    )
    assert actual == expected


@pytest.mark.parametrize(
    'expected, exclude_unmapped',
    [
        pytest.param(
            pd.DataFrame(
                [
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
                    {
                        'column_1': 'spore_germination',
                        'value_1': 'wild type',
                        'column_2': np.nan,
                        'value_2': np.nan,
                        'primary_id': 'PHIPO:0000956',
                        'primary_label': 'normal asexual spore germination frequency',
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'pathogen_genotype',
                        'extension_relation': np.nan,
                        'extension_range': np.nan,
                    },
                    {
                        'column_1': 'in_vitro_growth',
                        'value_1': 'reduced',
                        'column_2': 'is_filamentous',
                        'value_2': True,
                        'primary_id': 'PHIPO:0001212',
                        'primary_label': 'decreased hyphal growth',
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'pathogen_genotype',
                        'extension_relation': np.nan,
                        'extension_range': np.nan,
                    },
                    {
                        'column_1': 'vegetative_spores',
                        'value_1': 'increased',
                        'column_2': np.nan,
                        'value_2': np.nan,
                        'primary_id': 'PHIPO:0000051',
                        'primary_label': 'increased number of asexual spores',
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'pathogen_genotype',
                        'extension_relation': np.nan,
                        'extension_range': np.nan,
                    },
                    {
                        'column_1': 'mutant_phenotype',
                        'value_1': 'reduced virulence',
                        'column_2': np.nan,
                        'value_2': np.nan,
                        'primary_id': np.nan,
                        'primary_label': np.nan,
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'metagenotype',
                        'extension_relation': 'infective_ability',
                        'extension_range': 'PHIPO:0000015',
                    },
                ],
                index=[0, 2, 3, 4, 5, 6, 8],
                dtype='object',
            ),
            True,  # exclude_unmapped
            id='exclude_unmapped',
        ),
        pytest.param(
            pd.DataFrame(
                [
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
                        'value_2': False,
                        'primary_id': np.nan,
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
                    {
                        'column_1': 'spore_germination',
                        'value_1': 'wild type',
                        'column_2': np.nan,
                        'value_2': np.nan,
                        'primary_id': 'PHIPO:0000956',
                        'primary_label': 'normal asexual spore germination frequency',
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'pathogen_genotype',
                        'extension_relation': np.nan,
                        'extension_range': np.nan,
                    },
                    {
                        'column_1': 'in_vitro_growth',
                        'value_1': 'reduced',
                        'column_2': 'is_filamentous',
                        'value_2': True,
                        'primary_id': 'PHIPO:0001212',
                        'primary_label': 'decreased hyphal growth',
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'pathogen_genotype',
                        'extension_relation': np.nan,
                        'extension_range': np.nan,
                    },
                    {
                        'column_1': 'vegetative_spores',
                        'value_1': 'increased',
                        'column_2': np.nan,
                        'value_2': np.nan,
                        'primary_id': 'PHIPO:0000051',
                        'primary_label': 'increased number of asexual spores',
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'pathogen_genotype',
                        'extension_relation': np.nan,
                        'extension_range': np.nan,
                    },
                    {
                        'column_1': 'essential_gene',
                        'value_1': 'no',
                        'column_2': np.nan,
                        'value_2': np.nan,
                        'primary_id': np.nan,
                        'primary_label': 'viable population',
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'pathogen_genotype',
                        'extension_relation': np.nan,
                        'extension_range': np.nan,
                    },
                    {
                        'column_1': 'mutant_phenotype',
                        'value_1': 'reduced virulence',
                        'column_2': np.nan,
                        'value_2': np.nan,
                        'primary_id': np.nan,
                        'primary_label': np.nan,
                        'eco_id': np.nan,
                        'eco_label': np.nan,
                        'feature': 'metagenotype',
                        'extension_relation': 'infective_ability',
                        'extension_range': 'PHIPO:0000015',
                    },
                ],
                dtype='object',
            ),
            False,  # exclude_unmapped
            id='include_unmapped',
        ),
    ],
)
def test_load_phenotype_column_mapping(expected, exclude_unmapped):
    actual = loaders.load_phenotype_column_mapping(
        TEST_DATA_DIR / 'phenotype_mapping.csv', exclude_unmapped
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_load_bto_id_mapping():
    data = {
        'tissues, cell types and enzyme sources': 'BTO:0000000',
        'culture condition:-induced cell': 'BTO:0000001',
        'culture condition:1,4-dichlorobenzene-grown cell': 'BTO:0000002',
        'flower': 'BTO:0000469',
    }
    expected = pd.Series(data, name='term_id').rename_axis('term_label')
    actual = loaders.load_bto_id_mapping(TEST_DATA_DIR / 'bto.csv')
    pd.testing.assert_series_equal(actual, expected)


def test_load_phipo_mapping():
    expected = {
        'PHIPO:0000001': 'pathogen host interaction phenotype',
        'PHIPO:0000002': 'single species phenotype',
        'PHIPO:0000003': 'tissue phenotype',
        'PHIPO:0000004': 'unaffected pathogenicity',
    }
    actual = loaders.load_phipo_mapping(TEST_DATA_DIR / 'phipo.csv')
    assert actual == expected


def test_load_exp_tech_mapping():
    data = [
        {
            'exp_technique_stable': 'AA substitution with overexpression',
            'gene_protein_modification': 'D144P-Y222A',
            'exp_technique_transient': np.nan,
            'allele_type': 'amino acid substitution(s)',
            'name': np.nan,
            'description': 'D144P-Y222A',
            'expression': 'Overexpression',
            'evidence_code': 'Unknown',
            'eco': np.nan,
        },
        {
            'exp_technique_stable': 'AA substitution and deletion',
            'gene_protein_modification': 'ABCdelta, D144A-Y222A',
            'exp_technique_transient': np.nan,
            'allele_type': 'amino acid substitution(s) | deletion',
            'name': np.nan,
            'description': 'D144A,Y222A',
            'expression': 'Overexpression',
            'evidence_code': 'Unknown',
            'eco': np.nan,
        },
        {
            'exp_technique_stable': 'duplicate row',
            'gene_protein_modification': 'D144A-Y222A',
            'exp_technique_transient': np.nan,
            'allele_type': 'amino acid substitution(s)',
            'name': np.nan,
            'description': 'D144A,Y222A',
            'expression': 'Overexpression',
            'evidence_code': 'Unknown',
            'eco': np.nan,
        },
    ]
    expected = pd.DataFrame.from_records(data)
    actual = loaders.load_exp_tech_mapping(TEST_DATA_DIR / 'allele_mapping.csv')
    pd.testing.assert_frame_equal(actual, expected)


def test_load_chemical_data(chemical_data):
    actual = loaders.load_chemical_data(TEST_DATA_DIR / 'chemical_data.csv')
    expected = chemical_data
    assert actual == expected


def test_all_default_paths():
    DATA_DIR = importlib.resources.files('phibase_pipeline') / 'data'
    loaders_and_filenames = (
        (loaders.load_bto_id_mapping, ('bto.csv',)),
        (loaders.load_chemical_data, ('chemical_data.csv',)),
        (loaders.load_disease_column_mapping, ('phido.csv', 'disease_mapping.csv')),
        (loaders.load_exp_tech_mapping, ('allele_mapping.csv',)),
        (loaders.load_in_vitro_growth_classifier, ('in_vitro_growth_mapping.csv',)),
        (loaders.load_phenotype_column_mapping, ('phenotype_mapping.csv',)),
        (loaders.load_phipo_mapping, ('phipo.csv',)),
        (loaders.load_tissue_replacements, ('bto_renames.csv',)),
        (loaders.read_phig_uniprot_mapping, ('phig_uniprot_mapping.csv',)),
        (loaders.read_phipo_chebi_mapping, ('phipo_chebi_mapping.csv',)),
    )
    loaders_and_args = tuple(
        (
            (loader, tuple((DATA_DIR / filename for filename in filenames)))
            for loader, filenames in loaders_and_filenames
        )
    )
    for loader, args in loaders_and_args:
        actual = loader()
        expected = loader(*args)
        if isinstance(expected, pd.Series):
            assert_series_equal(actual, expected)
        elif isinstance(expected, pd.DataFrame):
            assert_frame_equal(actual, expected)
        else:
            assert actual == expected
