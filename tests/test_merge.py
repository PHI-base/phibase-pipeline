import pytest

from phibase_pipeline.merge import (
    update_session_ids,
)


def empty_session_dict():
    return {
        'PMID:1': {
            'phibase': {},
            'canto': {},
        },
    }


def session_with_id(session_id, pmid='PMID:1'):
    return {
        'genes': {
            'Fusarium graminearum I1S474': {
                'organism': 'Fusarium graminearum',
                'uniquename': 'I1S474',
            },
        },
        'alleles': {
            f'I1S474:{session_id}-1': {
                'allele_type': 'deletion',
                'gene': 'Fusarium graminearum I1S474',
                'name': 'ScOrtholog_PIG2delta',
                'primary_identifier': f'I1S474:{session_id}-1',
                'synonyms': [],
            },
        },
        'annotations': [
            {
                'checked': 'yes',
                'conditions': [],
                'creation_date': '2024-01-01',
                'evidence_code': '',
                'extension': [
                    {
                        'rangeDisplayName': 'reduced virulence',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000015',
                        'relation': 'infective_ability',
                    },
                    {
                        'rangeDisplayName': 'flower',
                        'rangeType': 'Ontology',
                        'rangeValue': 'BTO:0000469',
                        'relation': 'infects_tissue',
                    },
                ],
                'curator': {'community_curated': False},
                'figure': '',
                'phi4_id': ['PHI:1'],
                'publication': pmid,
                'status': 'new',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_host_interaction_phenotype',
                'metagenotype': f'{session_id}-metagenotype-1',
            },
            {
                'checked': 'yes',
                'conditions': [],
                'creation_date': '2024-01-01',
                'evidence_code': '',
                'extension': [
                    {
                        'rangeDisplayName': 'reduced virulence',
                        'rangeType': 'Ontology',
                        'rangeValue': 'PHIPO:0000015',
                        'relation': 'infective_ability',
                    },
                    {
                        'rangeDisplayName': 'flower',
                        'rangeType': 'Ontology',
                        'rangeValue': 'BTO:0000469',
                        'relation': 'infects_tissue',
                    },
                ],
                'curator': {'community_curated': False},
                'figure': '',
                'phi4_id': ['PHI:1'],
                'publication': pmid,
                'status': 'new',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_host_interaction_phenotype',
                'metagenotype': f'{session_id}-metagenotype-1',
            },
            {
                'checked': 'yes',
                'conditions': [],
                'creation_date': '2024-01-01',
                'evidence_code': '',
                'extension': [],
                'curator': {'community_curated': False},
                'figure': '',
                'phi4_id': ['PHI:1'],
                'publication': pmid,
                'status': 'new',
                'term': 'PHIPO:0000001',
                'type': 'pathogen_phenotype',
                'genotype': f'{session_id}-genotype-1',
            },
            {
                'checked': 'yes',
                'conditions': [],
                'creation_date': '2024-01-01',
                'evidence_code': '',
                'extension': [],
                'curator': {'community_curated': False},
                'figure': '',
                'phi4_id': ['PHI:1'],
                'publication': pmid,
                'status': 'new',
                'term': 'GO:0000001',
                'type': 'biological_process',
                'gene': 'Fusarium graminearum I1S474',
            },
        ],
        'genotypes': {
            f'{session_id}-genotype-1': {
                'loci': [[{'id': f'I1S474:{session_id}-1'}]],
                'organism_strain': 'PH-1',
                'organism_taxonid': 5518,
            }
        },
        'metadata': {
            'accepted_timestamp': '2024-01-01 00:00:00',
            'annotation_mode': 'advanced',
            'annotation_status': 'APPROVED',
            'annotation_status_datestamp': '2024-01-01 00:00:00',
            'approval_in_progress_timestamp': '2024-01-01 00:00:00',
            'approved_timestamp': '2024-01-01 00:00:00',
            'canto_session': f'{session_id}',
            'curation_accepted_date': '2024-01-01 00:00:00',
            'curation_in_progress_timestamp': '2024-01-01 00:00:00',
            'curation_pub_id': pmid,
            'curator_role': 'community',
            'first_approved_timestamp': '2024-01-01 00:00:00',
            'has_community_curation': True,
            'needs_approval_timestamp': '2024-01-01 00:00:00',
            'reactivated_timestamp': '2024-01-01 00:00:00',
            'session_created_timestamp': '2024-01-01 00:00:00',
            'session_first_submitted_timestamp': '2024-01-01 00:00:00',
            'session_genes_count': 1,
            'session_reactivated_timestamp': '2024-01-01 00:00:00',
            'session_term_suggestions_count': '0',
            'session_unknown_conditions_count': '0',
            'term_suggestion_count': '0',
            'unknown_conditions_count': '0',
        },
        'metagenotypes': {
            f'{session_id}-metagenotype-1': {
                'pathogen_genotype': f'{session_id}-genotype-1',
                'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                'type': 'pathogen-host',
            }
        },
        'organisms': {
            '5518': {'full_name': 'Fusarium graminearum'},
            '4565': {'full_name': 'Triticum aestivum'},
        },
        'publications': {
            pmid: {},
        },
    }


def export_with_ids(*pmids_and_session_ids):
    return {
        'curation_sessions': {
            session_id: session_with_id(session_id, pmid)
            for pmid, session_id in pmids_and_session_ids
        },
        'schema_version': 1,
    }


def session_dict_with_ids(*session_dict_data):
    return {
        pmid: {
            'phibase': session_with_id(session_id_a, pmid),
            'canto': session_with_id(session_id_b, pmid),
        }
        for pmid, session_id_a, session_id_b in session_dict_data
    }


@pytest.mark.parametrize(
    'sessions,expected',
    [
        pytest.param(
            empty_session_dict(),
            empty_session_dict(),
            id='empty',
        ),
        pytest.param(
            session_dict_with_ids(('PMID:1', '0123456789abcdef', 'abcdef0123456789')),
            session_dict_with_ids(('PMID:1', 'abcdef0123456789', 'abcdef0123456789')),
            id='all',
        ),
    ],
)
def test_update_session_ids(sessions, expected):
    actual = update_session_ids(sessions)
    assert expected == actual
