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
    make_ensembl_exports,
    read_phig_uniprot_mapping,
    read_phipo_chebi_mapping,
)

TEST_DATA_DIR = Path(__file__).parent / 'data'
ENSEMBL_DATA_DIR = TEST_DATA_DIR / 'ensembl'


@pytest.fixture
def phicanto_export():
    path = ENSEMBL_DATA_DIR / 'phicanto_export.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def effector_export():
    path = ENSEMBL_DATA_DIR / 'phicanto_export_effector.json'
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
    expected = pd.read_csv(ENSEMBL_DATA_DIR / 'get_canto_columns_expected.csv')
    effector_ids = set(['A0A507D1H9'])
    actual = get_canto_columns(phicanto_export, effector_ids)
    assert_frame_equal(expected, actual)


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


def test_combine_canto_uniprot_data():
    canto_data = pd.read_csv(ENSEMBL_DATA_DIR / 'get_canto_columns_expected.csv')
    uniprot_data = pd.read_csv(ENSEMBL_DATA_DIR / 'get_uniprot_columns_expected.csv')
    expected = pd.read_csv(
        ENSEMBL_DATA_DIR / 'combine_canto_uniprot_data_expected.csv',
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
    actual = make_ensembl_canto_export(export, uniprot_data)
    assert_frame_equal(expected, actual, check_dtype=False)


def test_make_ensembl_exports():
    test_data = ENSEMBL_DATA_DIR / 'make_ensembl_export'
    phi_df = pd.read_csv(test_data / 'phi_df.csv')
    canto_export = {
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
                    'conditions': ['PECO:0005241', 'PECO:0005322'],
                    'creation_date': '2023-06-30',
                    'curator': {'community_curated': True},
                    'evidence_code': 'Macroscopic observation (qualitative observation)',
                    'extension': [
                        {
                            'rangeDisplayName': 'spikelet',
                            'rangeType': 'Ontology',
                            'rangeValue': 'BTO:0002119',
                            'relation': 'infects_tissue',
                        },
                        {
                            'rangeDisplayName': 'disease present',
                            'rangeType': 'Ontology',
                            'rangeValue': 'PHIPO:0001200',
                            'relation': 'interaction_outcome',
                        },
                        {
                            'rangeDisplayName': 'FgSch9+[WT level] Fusarium graminearum (PH-1) / wild type Triticum aestivum (cv. Zimai 22)',
                            'rangeType': 'Metagenotype',
                            'rangeValue': '2d2e1c30cceb7aab-metagenotype-2',
                            'relation': 'compared_to_control',
                        },
                        {
                            'rangeDisplayName': 'reduced virulence',
                            'rangeType': 'Ontology',
                            'rangeValue': 'PHIPO:0000015',
                            'relation': 'infective_ability',
                        },
                    ],
                    'figure': 'Figure 3A',
                    'metagenotype': '2d2e1c30cceb7aab-metagenotype-1',
                    'publication': 'PMID:24903410',
                    'status': 'new',
                    'submitter_comment': 'Wheat heads were point-inoculated with conidial suspension of the wild-type PH-1 and FgSch9Δ mutant strains. Infected wheat heads were photographed 15 days after inoculation. Scab symptoms generated by ΔFgSch9 were restricted to the inoculated spikelets and those in the immediate vicinity, in comparison to widespread symptoms caused by the wild type strain, indicating FgSch9 is required for full virulence of F. graminearum.',
                    'term': 'PHIPO:0000985',
                    'type': 'pathogen_host_interaction_phenotype',
                },
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
                },
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
                },
                'Triticum-aestivum-wild-type-genotype-cv.-Zimai-22': {
                    'loci': [],
                    'organism_strain': 'cv. Zimai 22',
                    'organism_taxonid': 4565,
                },
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
    }
    uniprot_data = pd.read_csv(test_data / 'uniprot_data.tsv', sep='\t')
    expected = {
        k: pd.read_csv(test_data / filename)
        for k, filename in (
            ('phibase4', 'phibase4_expected.csv'),
            ('phibase5', 'phibase5_expected.csv'),
            ('amr', 'amr_expected.csv'),
        )
    }
    actual = make_ensembl_exports(phi_df, canto_export, uniprot_data)
    assert actual == expected


def test_read_phig_uniprot_mapping():
    path = TEST_DATA_DIR / 'phig_uniprot_mapping.csv'
    expected = {
        'A0A059ZHI3': 'PHIG:3546',
        'A0A059ZQI8': 'PHIG:1040',
        'A0A059ZR97': 'PHIG:3696',
    }
    actual = read_phig_uniprot_mapping(path)
    assert actual == expected


def test_read_phipo_chebi_mapping():
    path = ENSEMBL_DATA_DIR / 'phipo_chebi_mapping.csv'
    actual = read_phipo_chebi_mapping(path)
    expected = {
        'PHIPO:0000647': {'id': 'CHEBI:39214', 'label': 'abamectin'},
        'PHIPO:0000534': {'id': 'CHEBI:27666', 'label': 'actinomycin D'},
        'PHIPO:0000591': {'id': 'CHEBI:53661', 'label': 'alexidine'},
    }
    assert actual == expected
