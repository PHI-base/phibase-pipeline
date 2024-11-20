# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from phibase_pipeline import loaders
from phibase_pipeline.ensembl import (
    add_uniprot_columns,
    get_amr_records,
    get_canto_columns,
    get_effector_gene_ids,
    get_genotype_data,
    get_high_level_terms,
    get_metagenotype_data,
    get_physical_interaction_data,
    get_tissue_id_str,
    get_uniprot_columns,
    make_ensembl_amr_export,
    make_ensembl_canto_export,
    make_ensembl_exports,
    make_ensembl_phibase_export,
    merge_amr_uniprot_data,
    uniprot_data_to_mapping,
    write_ensembl_exports,
)

TEST_DATA_DIR = Path(__file__).parent / 'data'
ENSEMBL_DATA_DIR = TEST_DATA_DIR / 'ensembl'
ENSEMBL_EXPORT_DIR = ENSEMBL_DATA_DIR / 'make_ensembl_export'


@pytest.fixture
def canto_export():
    path = ENSEMBL_DATA_DIR / 'phicanto_export.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def canto_export_effector():
    path = ENSEMBL_DATA_DIR / 'phicanto_export_effector.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def canto_export2():
    """Simplified Canto export used for Ensembl export tests."""
    return {
        'curation_sessions': {
            '2d2e1c30cceb7aab': {
                'alleles': {
                    'I1RAE5:2d2e1c30cceb7aab-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RAE5',
                        'name': 'FG00472.1delta',
                        'primary_identifier': 'I1RAE5:2d2e1c30cceb7aab-1',
                        'synonyms': [],
                    }
                },
                'annotations': [
                    {
                        'checked': 'no',
                        'conditions': ['PECO:0000102', 'PECO:0005245', 'PECO:0005147'],
                        'creation_date': '2023-06-30',
                        'curator': {'community_curated': True},
                        'evidence_code': 'Cell growth assay',
                        'extension': [],
                        'figure': 'Figure 5A',
                        'genotype': '2d2e1c30cceb7aab-genotype-1',
                        'publication': 'PMID:24903410',
                        'status': 'new',
                        'submitter_comment': 'FgSch9Δ revealed increased resistance to the phenylpyrrole fungicide, fludioxonil, which targets the HOG pathway.',
                        'term': 'PHIPO:0000592',
                        'type': 'pathogen_phenotype',
                    }
                ],
                'genes': {
                    'Fusarium graminearum I1RAE5': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'I1RAE5',
                    }
                },
                'genotypes': {
                    '2d2e1c30cceb7aab-genotype-1': {
                        'loci': [[{'id': 'I1RAE5:2d2e1c30cceb7aab-1'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    }
                },
                'metadata': {
                    'accepted_timestamp': '2023-06-30 14:37:38',
                    'annotation_mode': 'advanced',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2024-03-05 12:04:59',
                    'approval_in_progress_timestamp': '2024-03-05 12:04:57',
                    'approved_timestamp': '2024-03-05 12:04:59',
                    'canto_session': '2d2e1c30cceb7aab',
                    'curation_accepted_date': '2023-06-30 14:37:38',
                    'curation_in_progress_timestamp': '2024-03-05 12:04:28',
                    'curation_pub_id': 'PMID:24903410',
                    'curator_role': 'community',
                    'first_approved_timestamp': '2024-03-05 12:01:36',
                    'has_community_curation': True,
                    'needs_approval_timestamp': '2024-03-05 12:04:53',
                    'reactivated_timestamp': '2024-03-05 12:04:28',
                    'session_created_timestamp': '2023-06-30 14:37:30',
                    'session_first_submitted_timestamp': '2024-02-28 14:03:17',
                    'session_genes_count': '1',
                    'session_reactivated_timestamp': '2024-03-05 12:04:28',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {},
                'organisms': {'5518': {'full_name': 'Fusarium graminearum'}},
                'publications': {'PMID:24903410': {}},
            }
        },
        'schema_version': 1,
    }


@pytest.fixture
def canto_export3():
    """Altered version of canto_export fixture used by test_make_ensembl_exports."""
    with open(ENSEMBL_EXPORT_DIR / 'canto_export3.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def canto_dataframe():
    return pd.read_csv(ENSEMBL_DATA_DIR / 'get_canto_columns_expected.csv')


def test_get_genotype_data(canto_export):
    session = canto_export['curation_sessions']['cc6cf06675cc6e13']
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


def test_get_metagenotype_data(canto_export):
    session = canto_export['curation_sessions']['cc6cf06675cc6e13']
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


def test_get_canto_columns(canto_export, canto_dataframe):
    effector_ids = set(['A0A507D1H9'])
    actual = get_canto_columns(canto_export, effector_ids)
    assert_frame_equal(canto_dataframe, actual)


def test_get_uniprot_columns():
    expected = pd.read_csv(
        ENSEMBL_DATA_DIR / 'get_uniprot_columns_expected.csv',
        dtype={
            'taxid_species': 'Int64',
            'taxid_strain': 'Int64',
        },
    )
    uniprot_data = pd.read_csv(ENSEMBL_DATA_DIR / 'uniprot_test_data.csv')
    actual = get_uniprot_columns(uniprot_data)
    assert_frame_equal(expected, actual)


def test_add_uniprot_columns(canto_dataframe):
    uniprot_data = pd.read_csv(ENSEMBL_DATA_DIR / 'get_uniprot_columns_expected.csv')
    expected = pd.read_csv(
        ENSEMBL_DATA_DIR / 'add_uniprot_columns_expected.csv',
        dtype={
            'phibase_id': 'object',
            'ensembl_a': 'object',
            'taxid_strain_a': 'Int64',
            'taxid_strain_b': 'Int64',
            'high_level_terms': 'object',
        },
    )
    actual = add_uniprot_columns(canto_dataframe, uniprot_data)
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


def test_get_effector_gene_ids(canto_export_effector):
    expected = set(('A0A507D1H9', 'A0A2K9YVY7', 'M4B6G6'))
    actual = get_effector_gene_ids(canto_export_effector)
    assert expected == actual


def test_get_physical_interaction_data():
    session_path = ENSEMBL_DATA_DIR / 'physical_interaction_session.json'
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


def test_make_ensembl_canto_export():
    expected = pd.read_csv(ENSEMBL_DATA_DIR / 'make_ensembl_canto_export_expected.csv')
    with open(ENSEMBL_DATA_DIR / 'phicanto_export.json', encoding='utf-8') as f:
        export = json.load(f)
    uniprot_data = pd.read_csv(ENSEMBL_DATA_DIR / 'uniprot_test_data.csv')
    phig_mapping = loaders.read_phig_uniprot_mapping(TEST_DATA_DIR / 'phig_uniprot_mapping.csv')
    actual = make_ensembl_canto_export(export, uniprot_data, phig_mapping)
    assert_frame_equal(expected, actual, check_dtype=False)


def test_make_ensembl_exports(canto_export3):
    phi_df = pd.read_csv(ENSEMBL_EXPORT_DIR / 'phi_df.csv')
    uniprot_data = pd.read_csv(ENSEMBL_EXPORT_DIR / 'uniprot_data.tsv', sep='\t')
    phig_mapping = loaders.read_phig_uniprot_mapping(TEST_DATA_DIR / 'phig_uniprot_mapping.csv')
    chebi_mapping = loaders.read_phipo_chebi_mapping(ENSEMBL_DATA_DIR / 'phipo_chebi_mapping.csv')

    in_vitro_growth_classifier = loaders.load_in_vitro_growth_classifier(
        TEST_DATA_DIR / 'in_vitro_growth_mapping.csv'
    )
    phenotype_mapping = loaders.load_phenotype_column_mapping(
        TEST_DATA_DIR / 'phenotype_mapping.csv'
    )
    disease_mapping = loaders.load_disease_column_mapping(
        phido_path=TEST_DATA_DIR / 'phido.csv',
        extra_path=TEST_DATA_DIR / 'disease_mapping.csv',
    )
    tissue_mapping = loaders.load_bto_id_mapping(TEST_DATA_DIR / 'bto.csv')
    exp_tech_mapping = pd.read_csv(ENSEMBL_DATA_DIR / 'phibase4_exp_tech_mapping.csv')

    expected = {
        k: pd.read_csv(ENSEMBL_EXPORT_DIR / filename)
        for k, filename in (
            ('phibase4', 'phibase4_expected.csv'),
            ('phibase5', 'phibase5_expected.csv'),
            ('amr', 'amr_expected.csv'),
        )
    }
    actual = make_ensembl_exports(
        phi_df,
        canto_export3,
        uniprot_data=uniprot_data,
        phig_mapping=phig_mapping,
        chebi_mapping=chebi_mapping,
        in_vitro_growth_classifier=in_vitro_growth_classifier,
        phenotype_mapping=phenotype_mapping,
        disease_mapping=disease_mapping,
        tissue_mapping=tissue_mapping,
        exp_tech_mapping=exp_tech_mapping,
    )
    assert actual.keys() == expected.keys()
    assert_frame_equal(actual['phibase4'], expected['phibase4'], check_dtype=False)
    assert_frame_equal(actual['phibase5'], expected['phibase5'], check_dtype=False)
    assert_frame_equal(actual['amr'], expected['amr'], check_dtype=False)


def test_make_ensembl_phibase_export():
    phi_df = pd.read_csv(ENSEMBL_EXPORT_DIR / 'phi_df.csv')
    uniprot_data = pd.read_csv(ENSEMBL_EXPORT_DIR / 'uniprot_data.tsv', sep='\t')
    expected = pd.read_csv(ENSEMBL_EXPORT_DIR / 'phibase4_expected.csv')

    in_vitro_growth_classifier = loaders.load_in_vitro_growth_classifier(
        TEST_DATA_DIR / 'in_vitro_growth_mapping.csv'
    )
    phenotype_mapping = loaders.load_phenotype_column_mapping(
        TEST_DATA_DIR / 'phenotype_mapping.csv'
    )
    disease_mapping = loaders.load_disease_column_mapping(
        phido_path=TEST_DATA_DIR / 'phido.csv',
        extra_path=TEST_DATA_DIR / 'disease_mapping.csv',
    )
    tissue_mapping = loaders.load_bto_id_mapping(TEST_DATA_DIR / 'bto.csv')
    exp_tech_mapping = pd.read_csv(ENSEMBL_DATA_DIR / 'phibase4_exp_tech_mapping.csv')

    actual = make_ensembl_phibase_export(
        phi_df,
        uniprot_data=uniprot_data,
        phenotype_mapping=phenotype_mapping,
        disease_mapping=disease_mapping,
        tissue_mapping=tissue_mapping,
        in_vitro_growth_classifier=in_vitro_growth_classifier,
        exp_tech_mapping=exp_tech_mapping,
    )
    assert_frame_equal(actual, expected, check_dtype=False)


def test_uniprot_data_to_mapping():
    uniprot_data = pd.read_csv(ENSEMBL_DATA_DIR / 'uniprot_test_data.csv')
    actual = uniprot_data_to_mapping(uniprot_data)
    expected = {
        'A0A0L0V3D9': {
            'ensembl': np.nan,
            'taxid_species': 27350,
            'taxid_strain': 1165861,
            'uniprot_matches': 'strain',
        },
        'A0A0L0W290': {
            'ensembl': np.nan,
            'taxid_species': 27350,
            'taxid_strain': 1165861,
            'uniprot_matches': 'strain',
        },
        'Q7X9A6': {
            'ensembl': (
                'TraesARI2D03G01168670.1; '
                'TraesCAD_scaffold_092989_01G000200.1; '
                'TraesCLE_scaffold_085676_01G000200.1; '
                'TraesCS2D02G214700.1; '
                'TraesCS2D03G0451200.1; '
                'TraesJAG2D03G01159270.1; '
                'TraesJUL2D03G01159000.1; '
                'TraesKAR2D01G0132480.1; '
                'TraesLAC2B03G00896920.1; '
                'TraesLAC2D03G01104360.1; '
                'TraesLDM2D03G01153380.1; '
                'TraesMAC2D03G01151020.1; '
                'TraesNOR2D03G01168840.1; '
                'TraesPAR_scaffold_090294_01G000200.1; '
                'TraesRN2D0100482800.1; '
                'TraesROB_scaffold_090335_01G000200.1; '
                'TraesSTA2D03G01141500.1; '
                'TraesSYM2D03G01167240.1; '
                'TraesWEE_scaffold_091601_01G000200.1'
            ),
            'taxid_species': 4565,
            'taxid_strain': None,
            'uniprot_matches': 'species',
        },
        'A0A507D1H9': {
            'ensembl': np.nan,
            'taxid_species': 286115,
            'taxid_strain': None,
            'uniprot_matches': 'species',
        },
        'G4MXR2': {
            'ensembl': 'MGG_08054T0',
            'taxid_species': 318829,
            'taxid_strain': 242507,
            'uniprot_matches': 'strain',
        },
        'Q5NBT9': {
            'ensembl': 'Os01t0254100-01',
            'taxid_species': 4530,
            'taxid_strain': 39947,
            'uniprot_matches': 'strain',
        },
    }
    assert actual == expected


@pytest.mark.parametrize(
    'annotation,expected',
    (
        pytest.param({}, None, id='none'),
        pytest.param(
            {
                'extension': [
                    {
                        'rangeValue': 'BTO:0000001',
                        'relation': 'infects_tissue',
                    }
                ]
            },
            'BTO:0000001',
            id='single',
        ),
        # Also test that only infects_tissue relations are included
        pytest.param(
            {
                'extension': [
                    {
                        'rangeValue': 'BTO:0000001',
                        'relation': 'infects_tissue',
                    },
                    {
                        'rangeValue': 'BTO:0000002',
                        'relation': 'infects_tissue',
                    },
                    {
                        'rangeValue': 'FOO:0000001',
                        'relation': 'other_relation',
                    },
                ]
            },
            'BTO:0000001; BTO:0000002',
            id='multi',
        ),
    )
)
def test_get_tissue_id_str(annotation, expected):
    actual = get_tissue_id_str(annotation)
    assert actual == expected


def test_get_amr_records(canto_export2):
    phig_mapping = loaders.read_phig_uniprot_mapping(TEST_DATA_DIR / 'phig_uniprot_mapping.csv')
    chebi_mapping = loaders.read_phipo_chebi_mapping(ENSEMBL_DATA_DIR / 'phipo_chebi_mapping.csv')
    expected = [
        {
            'phig_id': 'PHIG:11',
            'interactor_A_molecular_id': 'I1RAE5',
            'ensembl_a': None,
            'taxid_species_a': 5518,
            'organism_a': 'Fusarium graminearum',
            'taxid_strain_a': None,
            'strain_a': 'PH-1',
            'uniprot_matches_a': None,
            'modification_a': 'FG00472.1delta (deletion)',
            'interactor_B_molecular_id': 'CHEBI:81763',
            'ensembl_b': None,
            'taxid_species_b': None,
            'organism_b': 'fludioxonil',
            'taxid_strain_b': None,
            'strain_b': None,
            'uniprot_matches_b': None,
            'modification_b': None,
            'phenotype': 'PHIPO:0000592',
            'disease': None,
            'host_tissue': None,
            'evidence_code': 'Cell growth assay',
            'interaction_type': 'antimicrobial_interaction',
            'pmid': 24903410,
            'high_level_terms': None,
            'interactor_A_sequence': None,
            'interactor_B_sequence': None,
            'buffer_col': None,
        }
    ]
    actual = get_amr_records(canto_export2, phig_mapping, chebi_mapping)
    assert actual == expected


def test_merge_amr_uniprot_data():
    record = {
        'phig_id': 'PHIG:4721',
        'interactor_A_molecular_id': 'A0A384J5Y1',
        'ensembl_a': None,
        'taxid_species_a': 40559,
        'organism_a': 'Botrytis cinerea',
        'taxid_strain_a': None,
        'strain_a': 'Nj5-10',
        'uniprot_matches_a': None,
        'modification_a': 'Bos1(Q846stop) (nonsense mutation) [Not assayed]',
        'interactor_B_molecular_id': 'CHEBI:81763',
        'ensembl_b': None,
        'taxid_species_b': None,
        'organism_b': 'fludioxonil',
        'taxid_strain_b': None,
        'strain_b': None,
        'uniprot_matches_b': None,
        'modification_b': None,
        'phenotype': 'PHIPO:0000592',
        'disease': None,
        'host_tissue': None,
        'evidence_code': 'Cell growth assay',
        'interaction_type': 'antimicrobial_interaction',
        'pmid': 30686204,
        'high_level_terms': None,
        'interactor_A_sequence': None,
        'interactor_B_sequence': None,
        'buffer_col': None,
    }
    # Test that UniProtKB IDs not in uniprot_mapping are skipped
    unmatched_uniprot_record = record.copy()
    unmatched_uniprot_record['interactor_A_molecular_id'] = 'L2FD62'

    amr_records = [record, unmatched_uniprot_record]
    uniprot_mapping = {
        'A0A384J5Y1': {
            'ensembl': 'Bcin01g06260.3; Bcin01g06260.5; Bcin01g06260.6',
            'taxid_species': 40559,
            'taxid_strain': 332648,
            'uniprot_matches': 'strain',
        },
    }
    expected_record = amr_records[0].copy()
    expected_record['ensembl_a'] = 'Bcin01g06260.3; Bcin01g06260.5; Bcin01g06260.6'
    expected_record['taxid_species_a'] = 40559
    expected_record['taxid_strain_a'] = 332648
    expected_record['uniprot_matches_a'] = 'strain'
    expected = [expected_record]
    actual = merge_amr_uniprot_data(amr_records, uniprot_mapping)
    assert actual == expected


def test_make_ensembl_amr_export(canto_export3):
    chebi_mapping = {'PHIPO:0000592': {'id': 'CHEBI:81763', 'label': 'fludioxonil'}}
    phig_mapping = {'I1RAE5': 'PHIG:11'}
    uniprot_mapping = {
        'I1RAE5': {
            'ensembl': 'CEF72408',
            'taxid_species': 5518,
            'taxid_strain': 229533,
            'uniprot_matches': 'strain',
        }
    }
    expected = pd.DataFrame(
        {
            'phig_id': 'PHIG:11',
            'interactor_A_molecular_id': 'I1RAE5',
            'ensembl_a': 'CEF72408',
            'taxid_species_a': 5518,
            'organism_a': 'Fusarium graminearum',
            'taxid_strain_a': pd.Series(229533, dtype='Int64'),
            'strain_a': 'PH-1',
            'uniprot_matches_a': 'strain',
            'modification_a': 'FG00472.1delta (deletion)',
            'interactor_B_molecular_id': 'CHEBI:81763',
            'ensembl_b': np.nan,
            'taxid_species_b': np.nan,
            'organism_b': 'fludioxonil',
            'taxid_strain_b': np.nan,
            'strain_b': np.nan,
            'uniprot_matches_b': np.nan,
            'modification_b': np.nan,
            'phenotype': 'PHIPO:0000592',
            'disease': np.nan,
            'host_tissue': np.nan,
            'evidence_code': 'Cell growth assay',
            'interaction_type': 'antimicrobial_interaction',
            'pmid': 24903410,
            'high_level_terms': np.nan,
            'interactor_A_sequence': np.nan,
            'interactor_B_sequence': np.nan,
            'buffer_col': np.nan,
        },
        index=[0],
    )
    actual = make_ensembl_amr_export(
        canto_export3,
        chebi_mapping=chebi_mapping,
        phig_mapping=phig_mapping,
        uniprot_mapping=uniprot_mapping,
    )
    assert_frame_equal(actual, expected, check_dtype=False)


def test_write_ensembl_exports(tmpdir):
    write_ensembl_exports(
        phibase_path=ENSEMBL_EXPORT_DIR / 'phi_df.csv',
        canto_export_path=ENSEMBL_EXPORT_DIR / 'canto_export3.json',
        uniprot_data_path=ENSEMBL_EXPORT_DIR / 'uniprot_data.tsv',
        dir_path=tmpdir,
    )
    actual_expected_map = (
        (
            tmpdir / 'phibase4_interactions_export.csv',
            ENSEMBL_EXPORT_DIR / 'phibase4_expected.csv',
        ),
        (
            tmpdir / 'phibase5_interactions_export.csv',
            ENSEMBL_EXPORT_DIR / 'phibase5_expected.csv',
        ),
        (
            tmpdir / 'phibase_amr_export.csv',
            ENSEMBL_EXPORT_DIR / 'amr_expected.csv',
        ),
    )
    for actual_path, expected_path in actual_expected_map:
        actual, expected = (pd.read_csv(path) for path in (actual_path, expected_path))
        assert_frame_equal(actual, expected)
