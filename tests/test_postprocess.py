import copy

import pytest

from phibase_pipeline.postprocess import (
    allele_ids_of_genotype,
    merge_phi_canto_curation,
    merge_duplicate_alleles,
)


def test_allele_ids_of_genotype():
    genotype = {
        'loci': [
            [
                {
                    'expression': 'Not assayed',
                    'id': 'Q00000:0123456789abcdef-1',
                }
            ],
            [
                {
                    'expression': 'Not assayed',
                    'id': 'Q00000:0123456789abcdef-2',
                }
            ],
        ],
        'organism_strain': 'Unknown strain',
        'organism_taxonid': 562,
    }
    expected = ['Q00000:0123456789abcdef-1', 'Q00000:0123456789abcdef-2']
    actual = list(allele_ids_of_genotype(genotype))
    assert actual == expected


def test_merge_phi_canto_curation():
    # Normal merging: assert len(curation_sessions) > old_session_count should
    # be implicitly tested by this path
    phi_base_export = {
        'curation_sessions': {
            'abcdef0000000001': {
                'foo': 'bar',
            }
        }
    }
    phi_canto_export = {
        'curation_sessions': {
            'abcdef0000000002': {
                'foo': 'bar',
            }
        }
    }
    expected = {
        'curation_sessions': {
            'abcdef0000000001': {
                'foo': 'bar',
            },
            'abcdef0000000002': {
                'foo': 'bar',
            },
        }
    }

    merge_phi_canto_curation(phi_base_export, phi_canto_export)
    assert phi_base_export == expected

    # Merging sessions that already exist
    with pytest.raises(KeyError, match='session_id [0-9a-f]{16} already exists'):
        # Merging the same data again will trigger the exception
        merge_phi_canto_curation(phi_base_export, phi_canto_export)


def test_merge_duplicate_alleles():
    no_allele_key = {
        '0000000001abcdef': {},
    }
    empty_allele_object = {
        '0000000001abcdef': {
            'alleles': {},
        }
    }
    less_than_two_alleles = {
        '0000000001abcdef': {
            'alleles': {
                'Q00000:0123456789abcdef-1': {},
            }
        }
    }
    no_alleles_to_merge = {
        '0000000001abcdef': {
            'alleles': {
                'Q00000:0123456789abcdef-1': {
                    'gene': 'Homo sapiens Q000000',
                    'allele_type': 'deletion',
                    'name': 'ABC1',
                    'description': 'none',
                    'synonyms': [],
                },
                'Q00000:0123456789abcdef-2': {
                    'gene': 'Homo sapiens Q000000',
                    'allele_type': 'deletion',
                    'name': 'ABC2',
                    'description': 'none',
                    'synonyms': [],
                },
            },
            'genotypes': {
                'Q00000:0123456789abcdef-1': {
                    'loci': [
                        [
                            {
                                'expression': 'Not assayed',
                                'id': 'Q00000:0123456789abcdef-1',
                            }
                        ]
                    ],
                    'organism_strain': 'FOO1',
                    'organism_taxonid': 9606,
                },
                'Q00000:0123456789abcdef-2': {
                    'loci': [
                        [
                            {
                                'expression': 'Not assayed',
                                'id': 'Q00000:0123456789abcdef-2',
                            }
                        ]
                    ],
                    'organism_strain': 'FOO2',
                    'organism_taxonid': 9606,
                },
            },
        }
    }
    alleles_to_merge = {
        '0000000001abcdef': {
            'alleles': {
                'Q00000:0123456789abcdef-1': {
                    'gene': 'Homo sapiens Q000000',
                    'allele_type': 'deletion',
                    'name': 'ABC1',
                    'description': 'none',
                    'synonyms': ['alpha'],
                },
                'Q00000:0123456789abcdef-2': {
                    'gene': 'Homo sapiens Q000000',
                    'allele_type': 'deletion',
                    'name': 'ABC1',
                    'description': 'none',
                    'synonyms': ['beta'],
                },
            },
            'genotypes': {
                '0123456789abcdef-genotype-1': {
                    'loci': [
                        [
                            {
                                'expression': 'Not assayed',
                                'id': 'Q00000:0123456789abcdef-1',
                            }
                        ]
                    ],
                    'organism_strain': 'FOO1',
                    'organism_taxonid': 9606,
                },
                '0123456789abcdef-genotype-2': {
                    'loci': [
                        [
                            {
                                'expression': 'Not assayed',
                                'id': 'Q00000:0123456789abcdef-2',
                            }
                        ]
                    ],
                    'organism_strain': 'FOO2',
                    'organism_taxonid': 9606,
                },
            },
        }
    }
    merged = {
        '0000000001abcdef': {
            'alleles': {
                'Q00000:0123456789abcdef-1': {
                    'gene': 'Homo sapiens Q000000',
                    'allele_type': 'deletion',
                    'name': 'ABC1',
                    'description': 'none',
                    'synonyms': ['alpha', 'beta'],
                },
            },
            'genotypes': {
                '0123456789abcdef-genotype-1': {
                    'loci': [
                        [
                            {
                                'expression': 'Not assayed',
                                'id': 'Q00000:0123456789abcdef-1',
                            }
                        ]
                    ],
                    'organism_strain': 'FOO1',
                    'organism_taxonid': 9606,
                },
                '0123456789abcdef-genotype-2': {
                    'loci': [
                        [
                            {
                                'expression': 'Not assayed',
                                'id': 'Q00000:0123456789abcdef-1',
                            }
                        ]
                    ],
                    'organism_strain': 'FOO2',
                    'organism_taxonid': 9606,
                },
            },
        }
    }
    # Test that objects are not modified
    objects = (
        no_allele_key,
        empty_allele_object,
        less_than_two_alleles,
        no_alleles_to_merge,
    )
    for obj in objects:
        expected = obj
        actual = copy.deepcopy(obj)
        merge_duplicate_alleles(actual)
        assert actual == expected

    # Test that objects are merged
    merge_duplicate_alleles(alleles_to_merge)
    assert alleles_to_merge == merged
