import copy

import pytest

from phibase_pipeline.postprocess import (
    allele_ids_of_genotype,
    merge_phi_canto_curation,
    merge_duplicate_alleles,
    remove_allele_gene_names,
    remove_curator_orcids,
    remove_duplicate_annotations,
    remove_invalid_annotations,
    remove_invalid_genotypes,
    remove_invalid_metagenotypes,
    remove_orphaned_alleles,
    remove_orphaned_genes,
    remove_orphaned_organisms,
)


def assert_unchanged_after_mutation(func, *objects):
    for obj in objects:
        expected = obj
        actual = copy.deepcopy(obj)
        func(actual)
        assert actual == expected


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
    # Normal merging
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

    # There should be more curation sessions after merging than before
    with pytest.raises(AssertionError):
        merge_phi_canto_curation({'curation_sessions': {}}, {'curation_sessions': {}})

    # Merging sessions that already exist should raise an error
    with pytest.raises(KeyError, match='session_id [0-9a-f]{16} already exists'):
        # Merging the same data again to trigger the error
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


def test_remove_curator_orcids():
    # Should not modify sessions with no annotations
    no_annotations = {
        'curation_sessions': {
            '0123456789abcdef': {
                'foo': 'bar',
            }
        }
    }
    # Should not modify sessions with no ORCIDs
    no_curator_orcids = {
        'curation_sessions': {
            '0123456789abcdef': {
                'annotations': [
                    {
                        'curator': {
                            'community_curated': True,
                        },
                        'foo': 'bar',
                    }
                ]
            }
        }
    }
    for obj in (no_annotations, no_curator_orcids):
        expected = obj
        actual = copy.deepcopy(obj)
        remove_curator_orcids(actual)
        assert actual == expected

    # ORCIDs should be removed
    actual = {
        'curation_sessions': {
            '0123456789abcdef': {
                'annotations': [
                    {
                        'curator': {
                            'community_curated': True,
                            'curator_orcid': '0000-0002-1028-6941',
                        },
                        'foo': 'bar',
                    }
                ]
            }
        }
    }
    expected = {
        'curation_sessions': {
            '0123456789abcdef': {
                'annotations': [
                    {
                        'curator': {
                            'community_curated': True,
                        },
                        'foo': 'bar',
                    }
                ]
            }
        }
    }
    remove_curator_orcids(actual)
    assert actual == expected


def test_remove_invalid_annotations():
    # Test that the following objects are not modified
    no_annotations = {
        'annotations': [],
    }
    valid_annotations = {
        'genes': {
            'Homo sapiens Q000000': {},
        },
        'genotypes': {
            '0132456789abcdef-genotype-1': {},
        },
        'metagenotypes': {
            '0132456789abcdef-metagenotype-1': {},
        },
        'annotations': [
            {
                'gene': 'Homo sapiens Q000000',
            },
            {
                'genotype': '0132456789abcdef-genotype-1',
            },
            {
                'metagenotype': '0132456789abcdef-metagenotype-1',
            },
        ],
    }
    assert_unchanged_after_mutation(
        remove_invalid_annotations, no_annotations, valid_annotations
    )

    # Remove annotations with no features
    annotation_with_no_feature = {
        'annotations': [
            {
                'foo': 'bar',
            }
        ],
    }
    remove_invalid_annotations(annotation_with_no_feature)
    assert annotation_with_no_feature == {'annotations': []}

    # Remove annotations that reference non-existent features
    invalid_annotations = {
        'genes': {
            'Homo sapiens Q000000': {},
        },
        'genotypes': {
            '0132456789abcdef-genotype-1': {},
        },
        'metagenotypes': {
            '0132456789abcdef-metagenotype-1': {},
        },
        'annotations': [
            {
                'gene': 'Homo sapiens Q000001',
            },
            {
                'genotype': '0132456789abcdef-genotype-2',
            },
            {
                'metagenotype': '0132456789abcdef-metagenotype-2',
            },
        ],
    }
    expected = {
        'genes': {
            'Homo sapiens Q000000': {},
        },
        'genotypes': {
            '0132456789abcdef-genotype-1': {},
        },
        'metagenotypes': {
            '0132456789abcdef-metagenotype-1': {},
        },
        'annotations': [],
    }
    remove_invalid_annotations(invalid_annotations)
    assert invalid_annotations == expected


def test_remove_invalid_genotypes():
    valid = {
        'genes': {
            'Homo sapiens Q00000': {
                'organism': 'Homo sapiens',
                'uniquename': 'Q00000',
            }
        },
        'organisms': {
            '9606': {
                'full_name': 'Homo sapiens',
            }
        },
        'alleles': {
            'Q00000:0123456789abcdef-1': {
                'allele_type': 'deletion',
                'gene': 'Homo sapiens Q00000',
                'name': 'ABCdelta',
                'primary_identifier': "Q00000:0d2bc8d23a8667f4-1",
                'synonyms': [],
            }
        },
        'genotypes': {
            '0123456789abcdef-genotype-1': {
                'loci': [
                    [
                        {
                            'id': 'Q00000:0123456789abcdef-1',
                        }
                    ]
                ],
                'organism_strain': 'Unknown strain',
                'organism_taxonid': 9606,
            }
        },
    }
    assert_unchanged_after_mutation(remove_invalid_genotypes, valid)

    invalid = copy.deepcopy(valid)
    invalid['genotypes']['0123456789abcdef-genotype-1']['organism_taxonid'] = 9607

    expected = copy.deepcopy(valid)
    del expected['genotypes']['0123456789abcdef-genotype-1']
    remove_invalid_genotypes(invalid)
    assert invalid == expected


def test_remove_invalid_metagenotypes():
    valid = {
        'genotypes': {
            '0123465789abcdef-genotype-1': {},
            '0123465789abcdef-genotype-2': {},
        },
        'metagenotypes': {
            '01234568790abcdef-metagenotype-1': {
                'host_genotype': 'Homo-sapiens-wild-type-genotype-Unknown-strain',
                'pathogen_genotype': '0123465789abcdef-genotype-2',
                'type': 'pathogen-host',
            }
        },
    }
    assert_unchanged_after_mutation(remove_invalid_metagenotypes, valid)

    expected = copy.deepcopy(valid)
    del expected['metagenotypes']['01234568790abcdef-metagenotype-1']

    # Checks currently only apply to pathogen genotype
    mg_id = '01234568790abcdef-metagenotype-1'
    invalid_id = '0123465789abcdef-genotype-3'
    invalid = copy.deepcopy(valid)
    invalid['metagenotypes'][mg_id]['pathogen_genotype'] = invalid_id
    remove_invalid_metagenotypes(invalid)
    assert invalid == expected


def test_remove_orphaned_alleles():
    valid = {
        'alleles': {
            'Q00000:0123456789abcdef-1': {
                'allele_type': 'deletion',
                'gene': 'Homo sapiens Q00000',
                'name': 'ABCdelta',
                'primary_identifier': "Q00000:0d2bc8d23a8667f4-1",
                'synonyms': [],
            }
        },
        'genotypes': {
            '0123456789abcdef-genotype-1': {
                'loci': [
                    [
                        {
                            'id': 'Q00000:0123456789abcdef-1',
                        }
                    ]
                ],
                'organism_strain': 'Unknown strain',
                'organism_taxonid': 9606,
            }
        },
    }
    assert_unchanged_after_mutation(remove_orphaned_alleles, valid)

    expected = copy.deepcopy(valid)
    invalid = copy.deepcopy(valid)
    invalid['alleles']['Q00000:0123456789abcdef-2'] = {
        'allele_type': 'wild_type',
        'gene': 'Homo sapiens Q00000',
        'name': 'ABC+',
        'primary_identifier': "Q00000:0d2bc8d23a8667f4-2",
        'synonyms': [],
    }
    remove_orphaned_alleles(invalid)
    assert invalid == expected


def test_remove_orphaned_genes():
    gene = {
        'Homo sapiens Q00000': {
            'organism': 'Homo sapiens',
            'uniquename': 'Q00000',
        }
    }
    gene_in_allele = {
        'genes': gene,
        'alleles': {
            'Q00000:0123456789abcdef-1': {
                'gene': 'Homo sapiens Q00000',
            }
        },
    }
    gene_in_annotation = {
        'genes': gene,
        'alleles': {},
        'annotations': [
            {
                'gene': 'Homo sapiens Q00000',
            }
        ],
    }
    gene_in_annotation_extension = {
        'genes': gene,
        'alleles': {},
        'annotations': [
            {
                'genotype': '0123456789abcdef-genotype-1',
                'extension': [
                    {
                        'rangeValue': 'Q00000',
                    }
                ],
            }
        ],
    }
    assert_unchanged_after_mutation(
        remove_orphaned_genes,
        gene_in_allele,
        gene_in_annotation,
        gene_in_annotation_extension,
    )
    # Remove genes not used in alleles or annotations
    invalid = {
        'genes': gene,
        'alleles': {},
        'annotations': [],
    }
    expected = {
        'genes': {},
        'alleles': {},
        'annotations': [],
    }
    remove_orphaned_genes(invalid)
    assert invalid == expected


def test_remove_orphaned_organisms():
    organism = {
        '9606': {
            'full_name': 'Homo sapiens',
        }
    }
    in_gene = {
        'organisms': organism,
        'genes': {
            'Homo sapiens Q00000': {
                'organism': 'Homo sapiens',
                'uniquename': 'Q00000',
            }
        },
    }
    in_genotype = {
        'organisms': organism,
        'genes': {},
        'genotypes': {
            'Homo-sapiens-wild-type-genotype-Unknown-strain': {
                'organism_taxonid': 9606,
            }
        },
    }
    assert_unchanged_after_mutation(
        remove_orphaned_organisms,
        in_gene,
        in_genotype,
    )
    invalid = {
        'organisms': organism,
        'genes': {},
        'genotypes': {},
    }
    expected = {
        'organisms': {},
        'genes': {},
        'genotypes': {},
    }
    remove_orphaned_organisms(invalid)
    assert invalid == expected


def test_remove_duplicate_annotations():
    annotation_1 = {
        'checked': 'no',
        'conditions': [
            'PECO:0000001',
        ],
        'creation_date': '2000-01-01',
        'curator': {
            'community_curated': False,
        },
        'evidence_code': 'Substance quantification evidence',
        'extension': [],
        'figure': 'Fig 1',
        'genotype': '0123456789abcdef-genotype-1',
        'publication': 'PMID:1',
        'status': 'new',
        'submitter_comment': '',
        'term': 'PHIPO:0000001',
        'type': 'pathogen_host_interaction_phenotype',
    }
    annotation_2 = annotation_1.copy()
    duplicated = {
        'curation_sessions': {
            '0123456789abcdef': {
                'annotations': [
                    annotation_1,
                    annotation_2,
                ]
            }
        }
    }
    expected = {
        'curation_sessions': {
            '0123456789abcdef': {
                'annotations': [
                    annotation_1,
                ]
            }
        }
    }
    remove_duplicate_annotations(duplicated)
    assert duplicated == expected

    # Test that unique annotations are preserved; mutate values for all keys
    for k in annotation_2:
        if k == 'conditions':
            annotation_2[k][0] = 'foo'
        elif k == 'curator':
            annotation_2[k]['community_curated'] = True
        else:
            annotation_2[k] = 'foo'
        assert_unchanged_after_mutation(remove_duplicate_annotations, duplicated)

    # Test that PHI IDs are ignored when comparing
    annotation_2 = annotation_1.copy()
    annotation_2['phi4_id'] = ['PHI:1']
    duplicated['curation_sessions']['0123456789abcdef']['annotations'] = [
        annotation_1,
        annotation_2,
    ]
    remove_duplicate_annotations(duplicated)
    assert duplicated == expected


def test_remove_allele_gene_names():
    actual = {
        'alleles': {
            'Q00000:0123456789abcef-1': {
                'allele_type': 'wild_type',
                'gene': 'Homo sapiens Q00000',
                'name': 'ABC+',
                'gene_name': 'ABC',
                'primary_identifier': "Q00000:0123456789abcef-1",
                'synonyms': [],
            }
        }
    }
    expected = {
        'alleles': {
            'Q00000:0123456789abcef-1': {
                'allele_type': 'wild_type',
                'gene': 'Homo sapiens Q00000',
                'name': 'ABC+',
                'primary_identifier': "Q00000:0123456789abcef-1",
                'synonyms': [],
            }
        }
    }
    remove_allele_gene_names(actual)
    assert actual == expected
