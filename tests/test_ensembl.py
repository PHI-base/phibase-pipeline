import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from phibase_pipeline.ensembl import (
    combine_canto_uniprot_data,
    get_canto_columns,
    get_effector_gene_ids,
    get_genotype_data,
    get_high_level_terms,
    get_metagenotype_data,
    get_physical_interaction_data,
    get_uniprot_columns,
    make_ensembl_canto_export,
)


TEST_DATA_DIR = Path(__file__).parent / 'data' / 'ensembl'


@pytest.fixture
def phicanto_export():
    path = TEST_DATA_DIR / 'phicanto_export.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def effector_export():
    path = TEST_DATA_DIR / 'phicanto_export_effector.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def test_get_genotype_data(phicanto_export):
    session = phicanto_export['curation_sessions']['cc6cf06675cc6e13']
    genotype_id = 'cc6cf06675cc6e13-genotype-1'
    suffix = '_a'
    expected = {
        'uniprot_a': 'A0A0L0V3D9',
        'taxid_a': 27350,
        'organism_a': 'Puccinia striiformis',
        'strain_a': 'CYR32',
        'modification_a': (
            'Pst_12806deltaSP(1-23) (partial amino acid deletion) [Not assayed]'
        ),
    }
    actual = get_genotype_data(session, genotype_id, suffix)
    assert expected == actual


def test_get_metagenotype_data(phicanto_export):
    session = phicanto_export['curation_sessions']['cc6cf06675cc6e13']
    metagenotype_id = 'cc6cf06675cc6e13-metagenotype-1'
    expected = {
        'uniprot_a': 'A0A0L0V3D9',
        'taxid_a': 27350,
        'organism_a': 'Puccinia striiformis',
        'strain_a': 'CYR32',
        'modification_a': (
            'Pst_12806deltaSP(1-23) (partial amino acid deletion) [Not assayed]'
        ),
        'uniprot_b': None,
        'taxid_b': 4100,
        'organism_b': 'Nicotiana benthamiana',
        'strain_b': 'Unknown strain',
        'modification_b': None,
    }
    actual = get_metagenotype_data(session, metagenotype_id)
    assert expected == actual


def test_get_canto_columns(phicanto_export):
    expected = pd.read_csv(TEST_DATA_DIR / 'get_canto_columns_expected.csv')
    effector_ids = set(['A0A507D1H9'])
    actual = get_canto_columns(phicanto_export, effector_ids)
    assert_frame_equal(expected, actual)


def test_get_uniprot_columns():
    expected = pd.read_csv(
        TEST_DATA_DIR / 'get_uniprot_columns_expected.csv',
        dtype={
            'taxid_species': 'Int64',
            'taxid_strain': 'Int64',
        },
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
        },
    )
    actual = combine_canto_uniprot_data(canto_data, uniprot_data)
    assert_frame_equal(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    'annotation,expected',
    (
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000022',
                'type': 'pathogen_phenotype',
            },
            ['Resistance to chemical'],
            id='resistance_to_chemical',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000021',
                'type': 'pathogen_phenotype',
            },
            ['Sensitivity to chemical'],
            id='sensitivity_to_chemical',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000513',
                'type': 'pathogen_phenotype',
            },
            ['Lethal'],
            id='lethal',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [
                    {
                        'rangeDisplayName': 'loss of mutualism',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000207',
                        'relation': 'infective_ability',
                    },
                ],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_phenotype',
            },
            ['Loss of mutualism'],
            id='loss_of_mutualism',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [
                    {
                        'rangeDisplayName': 'increased virulence',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000014',
                        'relation': 'infective_ability',
                    },
                ],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_phenotype',
            },
            ['Increased virulence'],
            id='increased_virulence',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [
                    {
                        'rangeDisplayName': 'loss of pathogenicity',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000010',
                        'relation': 'infective_ability',
                    },
                ],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_phenotype',
            },
            ['Loss of pathogenicity'],
            id='loss_of_pathogenicity',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [
                    {
                        'rangeDisplayName': 'reduced virulence',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000015',
                        'relation': 'infective_ability',
                    },
                ],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_phenotype',
            },
            ['Reduced virulence'],
            id='reduced_virulence',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [
                    {
                        'rangeDisplayName': 'unaffected pathogenicity',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000004',
                        'relation': 'infective_ability',
                    },
                ],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_phenotype',
            },
            ['Unaffected pathogenicity'],
            id='unaffected_pathogenicity',
        ),
        pytest.param(
            {
                'checked': 'no',
                'conditions': [],
                'creation_date': '2024-06-26',
                'curator': {'community_curated': True},
                'evidence_code': 'Cell growth assay',
                'extension': [
                    {
                        'rangeDisplayName': 'reduced virulence',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000015',
                        'relation': 'infective_ability',
                    },
                ],
                'genotype': '0123456789abcdef-genotype-1',
                'publication': 'PMID:1234567',
                'status': 'new',
                'submitter_comment': '',
                'term': 'PHIPO:0000021',
                'type': 'pathogen_phenotype',
            },
            ['Reduced virulence', 'Sensitivity to chemical'],
            id='primary_and_ext',
        ),
    ),
)
def test_get_high_level_terms(annotation, expected):
    actual = get_high_level_terms(annotation)
    assert expected == actual


def test_get_effector_gene_ids(effector_export):
    expected = set(('A0A507D1H9', 'A0A2K9YVY7', 'M4B6G6'))
    actual = get_effector_gene_ids(effector_export)
    assert expected == actual


def test_get_physical_interaction_data():
    session_path = TEST_DATA_DIR / 'physical_interaction_session.json'
    with open(session_path, encoding='utf-8') as f:
        session = json.load(f)
    expected = {
        'uniprot_a': 'G4MXR2',
        'taxid_a': 318829,
        'organism_a': 'Magnaporthe oryzae',
        'strain_a': None,
        'modification_a': None,
        'uniprot_b': 'Q5NBT9',
        'taxid_b': 4530,
        'organism_b': 'Oryza sativa',
        'strain_b': None,
        'modification_b': None,
    }
    gene_id = 'Magnaporthe oryzae G4MXR2'
    interacting_gene_id = 'Oryza sativa Q5NBT9'
    actual = get_physical_interaction_data(session, gene_id, interacting_gene_id)
    assert expected == actual


def test_make_ensembl_canto_export(tmpdir):
    expected = pd.read_csv(TEST_DATA_DIR / 'make_ensembl_canto_export_expected.csv')
    export_path = TEST_DATA_DIR / 'phicanto_export.json'
    uniprot_data_path = TEST_DATA_DIR / 'uniprot_test_data.csv'
    out_path = tmpdir / 'ensembl_canto_export.csv'
    make_ensembl_canto_export(export_path, uniprot_data_path, out_path)
    actual = pd.read_csv(out_path)
    assert_frame_equal(expected, actual, check_dtype=False)
