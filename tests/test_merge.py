import pytest

from phibase_pipeline.merge import (
    get_recurated_sessions,
    merge_recurated_sessions,
    rekey_duplicate_feature_ids,
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


def test_get_recurated_sessions():
    phibase_export = export_with_ids(
        ('PMID:1', '0000000001abcdef'),
        ('PMID:2', '0000000003abcdef'),
        ('PMID:3', '0000000005abcdef'),
    )
    canto_export = export_with_ids(
        ('PMID:1', '0000000002abcdef'),
        ('PMID:2', '0000000004abcdef'),
        ('PMID:4', '0000000006abcdef'),
    )
    expected = session_dict_with_ids(
        ('PMID:1', '0000000001abcdef', '0000000002abcdef'),
        ('PMID:2', '0000000003abcdef', '0000000004abcdef'),
    )
    actual = get_recurated_sessions(phibase_export, canto_export)
    assert expected == actual


@pytest.mark.parametrize(
    'feature_type, phibase_session, canto_session, expected',
    [
        pytest.param(
            # feature_type
            'alleles',
            # phibase_session
            {
                'alleles': {
                    'I1RYS3:0000000001abcdef-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'ScOrtholog_MET22delta',
                        'primary_identifier': 'I1RYS3:0000000001abcdef-1',
                        'synonyms': [],
                    },
                    'I1RYS3:0000000001abcdef-2': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'ScOrtholog_MET22delta2',
                        'primary_identifier': 'I1RYS3:0000000001abcdef-2',
                        'synonyms': [],
                    },
                    'I1RWQ1:0000000001abcdef-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RWQ1',
                        'name': 'ScOrtholog_INP53delta',
                        'primary_identifier': 'I1RWQ1:0000000001abcdef-1',
                        'synonyms': [],
                    },
                }
            },
            # canto_session
            {
                'alleles': {
                    # duplicate allele with unique name: use this name
                    'I1RYS3:0000000001abcdef-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'FG09532.1delta',
                        'primary_identifier': 'I1RYS3:0000000001abcdef-1',
                        'synonyms': [],
                    },
                    # unique allele with duplicate ID: renumber to 3
                    'I1RYS3:0000000001abcdef-2': {
                        'allele_type': 'wild_type',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'FG09532.1+',
                        'primary_identifier': 'I1RYS3:0000000001abcdef-2',
                        'synonyms': [],
                    },
                    # unique allele with duplicate ID: renumber to 4
                    'I1RWQ1:0000000001abcdef-1': {
                        'allele_type': 'wild_type',
                        'gene': 'Fusarium graminearum I1RWQ1',
                        'name': 'FG09532.1+',
                        'primary_identifier': 'I1RWQ1:0000000001abcdef-1',
                        'synonyms': [],
                    },
                    # unique allele with unique ID: do nothing
                    'I1RWQ1:0000000001abcdef-2': {
                        'allele_type': 'amino_acid_substitution',
                        'gene': 'Fusarium graminearum I1RWQ1',
                        'name': 'FG09532.1AA',
                        'primary_identifier': 'I1RWQ1:0000000001abcdef-2',
                        'synonyms': [],
                    },
                }
            },
            # expected
            {
                'I1RYS3:0000000001abcdef-1': {
                    'allele_type': 'deletion',
                    'gene': 'Fusarium graminearum I1RYS3',
                    'name': 'FG09532.1delta',
                    'primary_identifier': 'I1RYS3:0000000001abcdef-1',
                    'synonyms': [],
                },
                'I1RYS3:0000000001abcdef-3': {
                    'allele_type': 'wild_type',
                    'gene': 'Fusarium graminearum I1RYS3',
                    'name': 'FG09532.1+',
                    'primary_identifier': 'I1RYS3:0000000001abcdef-3',
                    'synonyms': [],
                },
                'I1RWQ1:0000000001abcdef-2': {
                    'allele_type': 'amino_acid_substitution',
                    'gene': 'Fusarium graminearum I1RWQ1',
                    'name': 'FG09532.1AA',
                    'primary_identifier': 'I1RWQ1:0000000001abcdef-2',
                    'synonyms': [],
                },
                'I1RWQ1:0000000001abcdef-4': {
                    'allele_type': 'wild_type',
                    'gene': 'Fusarium graminearum I1RWQ1',
                    'name': 'FG09532.1+',
                    'primary_identifier': 'I1RWQ1:0000000001abcdef-4',
                    'synonyms': [],
                },
            },
            id='alleles',
        ),
        pytest.param(
            # feature_type
            'genotypes',
            # phibase_session
            {
                'genotypes': {
                    '0000000001abcdef-genotype-1': {
                        'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-1'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                    '0000000001abcdef-genotype-2': {
                        'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-1'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                }
            },
            # canto_session
            {
                'genotypes': {
                    '0000000001abcdef-genotype-1': {
                        'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-1'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                    '0000000001abcdef-genotype-2': {
                        'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-2'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                }
            },
            # expected
            {
                '0000000001abcdef-genotype-3': {
                    'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-2'}]],
                    'organism_strain': 'PH-1',
                    'organism_taxonid': 5518,
                },
            },
            id='genotypes',
        ),
        pytest.param(
            # feature_type
            'metagenotypes',
            # phibase_session
            {
                'metagenotypes': {
                    '0000000001abcdef-metagenotype-1': {
                        'pathogen_genotype': '0000000001abcdef-genotype-1',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '0000000001abcdef-metagenotype-2': {
                        'pathogen_genotype': '0000000001abcdef-genotype-2',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
            },
            # canto_session
            {
                'metagenotypes': {
                    '0000000001abcdef-metagenotype-1': {
                        'pathogen_genotype': '0000000001abcdef-genotype-3',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '0000000001abcdef-metagenotype-2': {
                        'pathogen_genotype': '0000000001abcdef-genotype-2',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                }
            },
            # expected
            {
                '0000000001abcdef-metagenotype-3': {
                    'pathogen_genotype': '0000000001abcdef-genotype-3',
                    'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                    'type': 'pathogen-host',
                },
            },
            id='metagenotypes',
        ),
    ],
)
def test_rekey_duplicate_feature_ids(
    feature_type, phibase_session, canto_session, expected
):
    actual = rekey_duplicate_feature_ids(feature_type, phibase_session, canto_session)
    assert expected == actual


def test_merge_recurated_sessions():
    recurated_sessions = {
        'PMID:1': {
            'phibase': {
                'alleles': {
                    'I1RYS3:0000000002abcdef-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'ScOrtholog_MET22delta',
                        'primary_identifier': 'I1RYS3:0000000002abcdef-1',
                        'synonyms': [],
                    },
                    'I1RYS3:0000000002abcdef-2': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'ScOrtholog_MET22delta2',
                        'primary_identifier': 'I1RYS3:0000000002abcdef-2',
                        'synonyms': [],
                    },
                    'I1RWQ1:0000000002abcdef-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RWQ1',
                        'name': 'ScOrtholog_INP53delta',
                        'primary_identifier': 'I1RWQ1:0000000002abcdef-1',
                        'synonyms': [],
                    },
                },
                'annotations': [],
                'genes': {
                    'Fusarium graminearum I1RWQ1': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'I1RWQ1',
                    },
                    'Fusarium graminearum I1RYS3': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'I1RYS3',
                    },
                },
                'genotypes': {
                    '0000000002abcdef-genotype-1': {
                        'loci': [[{'id': 'A0A098DXK5:0000000002abcdef-1'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                    '0000000002abcdef-genotype-2': {
                        'loci': [[{'id': 'A0A098DXK5:0000000002abcdef-1'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                },
                'metadata': {
                    'accepted_timestamp': '2024-01-01 00:00:00',
                    'annotation_mode': 'advanced',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2024-01-01 00:00:00',
                    'approval_in_progress_timestamp': '2024-01-01 00:00:00',
                    'approved_timestamp': '2024-01-01 00:00:00',
                    'canto_session': '0000000002abcdef',
                    'curation_accepted_date': '2024-01-01 00:00:00',
                    'curation_in_progress_timestamp': '2024-01-01 00:00:00',
                    'curation_pub_id': 'PMID:1',
                    'curator_role': 'community',
                    'first_approved_timestamp': '2024-01-01 00:00:00',
                    'has_community_curation': True,
                    'needs_approval_timestamp': '2024-01-01 00:00:00',
                    'reactivated_timestamp': '2024-01-01 00:00:00',
                    'session_created_timestamp': '2024-01-01 00:00:00',
                    'session_first_submitted_timestamp': '2024-01-01 00:00:00',
                    'session_genes_count': 2,
                    'session_reactivated_timestamp': '2024-01-01 00:00:00',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {
                    '0000000002abcdef-metagenotype-1': {
                        'pathogen_genotype': '0000000002abcdef-genotype-1',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '0000000002abcdef-metagenotype-2': {
                        'pathogen_genotype': '0000000002abcdef-genotype-2',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
                'organisms': {
                    '5518': {'full_name': 'Fusarium graminearum'},
                    '4565': {'full_name': 'Triticum aestivum'},
                    '9606': {'full_name': 'Homo sapiens'},
                },
                'publications': {
                    'PMID:1': {},
                },
            },
            'canto': {
                'alleles': {
                    'I1RYS3:0000000001abcdef-1': {
                        'allele_type': 'deletion',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'FG09532.1delta',
                        'primary_identifier': 'I1RYS3:0000000001abcdef-1',
                        'synonyms': [],
                    },
                    'I1RYS3:0000000001abcdef-2': {
                        'allele_type': 'wild_type',
                        'gene': 'Fusarium graminearum I1RYS3',
                        'name': 'FG09532.1+',
                        'primary_identifier': 'I1RYS3:0000000001abcdef-2',
                        'synonyms': [],
                    },
                    'I1RWQ1:0000000001abcdef-1': {
                        'allele_type': 'wild_type',
                        'gene': 'Fusarium graminearum I1RWQ1',
                        'name': 'FG09532.1+',
                        'primary_identifier': 'I1RWQ1:0000000001abcdef-1',
                        'synonyms': [],
                    },
                    'I1RWQ1:0000000001abcdef-2': {
                        'allele_type': 'amino_acid_substitution',
                        'gene': 'Fusarium graminearum I1RWQ1',
                        'name': 'FG09532.1AA',
                        'primary_identifier': 'I1RWQ1:0000000001abcdef-2',
                        'synonyms': [],
                    },
                },
                'annotations': [],
                'genes': {
                    'Fusarium graminearum I1RWQ1': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'I1RWQ1',
                    },
                    'Fusarium graminearum I1RYS3': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'I1RYS3',
                    },
                    # test adding new gene
                    'Fusarium graminearum Q00909': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'Q00909',
                    },
                },
                'genotypes': {
                    '0000000001abcdef-genotype-1': {
                        'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-1'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                    '0000000001abcdef-genotype-2': {
                        'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-2'}]],
                        'organism_strain': 'PH-1',
                        'organism_taxonid': 5518,
                    },
                },
                'metadata': {
                    'accepted_timestamp': '2024-01-02 00:00:00',
                    'annotation_mode': 'advanced',
                    'annotation_status': 'APPROVED',
                    'annotation_status_datestamp': '2024-01-02 00:00:00',
                    'approval_in_progress_timestamp': '2024-01-02 00:00:00',
                    'approved_timestamp': '2024-01-02 00:00:00',
                    'canto_session': '0000000001abcdef',
                    'curation_accepted_date': '2024-01-02 00:00:00',
                    'curation_in_progress_timestamp': '2024-01-02 00:00:00',
                    'curation_pub_id': 'PMID:1',
                    'curator_role': 'community',
                    'first_approved_timestamp': '2024-01-02 00:00:00',
                    'has_community_curation': True,
                    'needs_approval_timestamp': '2024-01-02 00:00:00',
                    'reactivated_timestamp': '2024-01-02 00:00:00',
                    'session_created_timestamp': '2024-01-02 00:00:00',
                    'session_first_submitted_timestamp': '2024-01-02 00:00:00',
                    'session_genes_count': 2,
                    'session_reactivated_timestamp': '2024-01-02 00:00:00',
                    'session_term_suggestions_count': '0',
                    'session_unknown_conditions_count': '0',
                    'term_suggestion_count': '0',
                    'unknown_conditions_count': '0',
                },
                'metagenotypes': {
                    '0000000001abcdef-metagenotype-1': {
                        'pathogen_genotype': '0000000001abcdef-genotype-3',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                    '0000000001abcdef-metagenotype-2': {
                        'pathogen_genotype': '0000000001abcdef-genotype-2',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                        'type': 'pathogen-host',
                    },
                },
                'organisms': {
                    '5518': {'full_name': 'Fusarium graminearum'},
                    '4565': {'full_name': 'Triticum aestivum'},
                },
                'publications': {
                    'PMID:1': {},
                },
            },
        },
        'PMID:2': {
            'phibase': {},
            'canto': {},
        },
    }
    expected = {
        'PMID:1': {
            'alleles': {
                # from phibase, renamed by canto
                'I1RYS3:0000000001abcdef-1': {
                    'allele_type': 'deletion',
                    'gene': 'Fusarium graminearum I1RYS3',
                    'name': 'FG09532.1delta',
                    'primary_identifier': 'I1RYS3:0000000001abcdef-1',
                    'synonyms': [],
                },
                # from phibase
                'I1RYS3:0000000001abcdef-2': {
                    'allele_type': 'deletion',
                    'gene': 'Fusarium graminearum I1RYS3',
                    'name': 'ScOrtholog_MET22delta2',
                    'primary_identifier': 'I1RYS3:0000000001abcdef-2',
                    'synonyms': [],
                },
                # from phibase
                'I1RWQ1:0000000001abcdef-1': {
                    'allele_type': 'deletion',
                    'gene': 'Fusarium graminearum I1RWQ1',
                    'name': 'ScOrtholog_INP53delta',
                    'primary_identifier': 'I1RWQ1:0000000001abcdef-1',
                    'synonyms': [],
                },
                # from canto, rekeyed from 2
                'I1RYS3:0000000001abcdef-3': {
                    'allele_type': 'wild_type',
                    'gene': 'Fusarium graminearum I1RYS3',
                    'name': 'FG09532.1+',
                    'primary_identifier': 'I1RYS3:0000000001abcdef-3',
                    'synonyms': [],
                },
                # from canto, rekeyed from 1
                'I1RWQ1:0000000001abcdef-4': {
                    'allele_type': 'wild_type',
                    'gene': 'Fusarium graminearum I1RWQ1',
                    'name': 'FG09532.1+',
                    'primary_identifier': 'I1RWQ1:0000000001abcdef-4',
                    'synonyms': [],
                },
                # from canto
                'I1RWQ1:0000000001abcdef-2': {
                    'allele_type': 'amino_acid_substitution',
                    'gene': 'Fusarium graminearum I1RWQ1',
                    'name': 'FG09532.1AA',
                    'primary_identifier': 'I1RWQ1:0000000001abcdef-2',
                    'synonyms': [],
                },
            },
            'annotations': [],
            'genes': {
                # from phibase, duplicated in canto
                'Fusarium graminearum I1RWQ1': {
                    'organism': 'Fusarium graminearum',
                    'uniquename': 'I1RWQ1',
                },
                # from phibase, duplicated in canto
                'Fusarium graminearum I1RYS3': {
                    'organism': 'Fusarium graminearum',
                    'uniquename': 'I1RYS3',
                },
                # from canto
                'Fusarium graminearum Q00909': {
                    'organism': 'Fusarium graminearum',
                    'uniquename': 'Q00909',
                },
            },
            'genotypes': {
                # from phibase, duplicated in canto
                '0000000001abcdef-genotype-1': {
                    'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-1'}]],
                    'organism_strain': 'PH-1',
                    'organism_taxonid': 5518,
                },
                # from phibase
                '0000000001abcdef-genotype-2': {
                    'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-1'}]],
                    'organism_strain': 'PH-1',
                    'organism_taxonid': 5518,
                },
                # from canto, rekeyed from 2
                '0000000001abcdef-genotype-3': {
                    'loci': [[{'id': 'A0A098DXK5:0000000001abcdef-2'}]],
                    'organism_strain': 'PH-1',
                    'organism_taxonid': 5518,
                },
            },
            'metadata': {
                # dates updated by canto
                'accepted_timestamp': '2024-01-02 00:00:00',
                'annotation_mode': 'advanced',
                'annotation_status': 'APPROVED',
                'annotation_status_datestamp': '2024-01-02 00:00:00',
                'approval_in_progress_timestamp': '2024-01-02 00:00:00',
                'approved_timestamp': '2024-01-02 00:00:00',
                'canto_session': '0000000001abcdef',
                'curation_accepted_date': '2024-01-02 00:00:00',
                'curation_in_progress_timestamp': '2024-01-02 00:00:00',
                'curation_pub_id': 'PMID:1',
                'curator_role': 'community',
                'first_approved_timestamp': '2024-01-02 00:00:00',
                'has_community_curation': True,
                'needs_approval_timestamp': '2024-01-02 00:00:00',
                'reactivated_timestamp': '2024-01-02 00:00:00',
                'session_created_timestamp': '2024-01-02 00:00:00',
                'session_first_submitted_timestamp': '2024-01-02 00:00:00',
                'session_genes_count': 3,  # updated by merging
                'session_reactivated_timestamp': '2024-01-02 00:00:00',
                'session_term_suggestions_count': '0',
                'session_unknown_conditions_count': '0',
                'term_suggestion_count': '0',
                'unknown_conditions_count': '0',
            },
            'metagenotypes': {
                # from phibase
                '0000000001abcdef-metagenotype-1': {
                    'pathogen_genotype': '0000000001abcdef-genotype-1',
                    'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                    'type': 'pathogen-host',
                },
                # from phibase, duplicated in canto
                '0000000001abcdef-metagenotype-2': {
                    'pathogen_genotype': '0000000001abcdef-genotype-2',
                    'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                    'type': 'pathogen-host',
                },
                # from canto, rekeyed from 1
                '0000000001abcdef-metagenotype-3': {
                    'pathogen_genotype': '0000000001abcdef-genotype-3',
                    'host_genotype': 'Triticum-aestivum-wild-type-genotype-Unknown-strain',
                    'type': 'pathogen-host',
                },
            },
            'organisms': {
                '5518': {'full_name': 'Fusarium graminearum'},
                '4565': {'full_name': 'Triticum aestivum'},
                # from phibase
                '9606': {'full_name': 'Homo sapiens'},
            },
            'publications': {
                'PMID:1': {},
            },
        },
        'PMID:2': {},
    }
    actual = merge_recurated_sessions(recurated_sessions)
    assert expected == actual
