# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import copy
import json
from pathlib import Path

import pytest

from phibase_pipeline.postprocess import (
    add_chemical_extensions,
    add_delta_symbol,
    add_proteome_strains_to_genes,
    add_pubmed_data_to_sessions,
    add_uniprot_data_to_genes,
    allele_ids_of_genotype,
    get_all_pmids_in_export,
    get_all_uniprot_ids_in_export,
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
    remove_unapproved_sessions,
    truncate_phi4_ids,
)

DATA_DIR = Path(__file__).parent / 'data'
UNIPROT_DATA_DIR = DATA_DIR / 'uniprot'


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


@pytest.fixture
def export_for_xrefs():
    return {
        'curation_sessions': {
            '0123456789abcdef': {
                'alleles': {},
                'annotations': [],
                'genes': {
                    'Fusarium graminearum Q00909': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'Q00909',
                    }
                },
                'genotypes': {},
                'metadata': {},
                'metagenotypes': {},
                'organisms': {},
                'publications': {'PMID:12345678': {}},
            }
        },
        'schema_version': 1,
    }


@pytest.fixture
def export_with_uniprot_data(export_for_xrefs):
    uniprot_gene_data = {
        'Q00909': {
            'uniprot_id': 'Q00909',
            'name': 'TRI5',
            'product': 'Trichodiene synthase',
            'strain': 'Gibberella zeae (strain ATCC MYA-4620 / CBS 123657 / FGSC 9075 / NRRL 31084 / PH-1)',
            'dbref_gene_id': '23550840',
            'ensembl_sequence_id': [],
            'ensembl_gene_id': [],
        }
    }
    return add_uniprot_data_to_genes(export_for_xrefs, uniprot_gene_data)


@pytest.fixture
def proteome_results():
    path = UNIPROT_DATA_DIR / 'proteome_results_UP000070720.json'
    with open(path, encoding='utf-8') as json_file:
        return json.load(json_file)


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


def test_add_delta_symbol():
    template = {
        'alleles': {
            'Q00000:0123456789abcef-1': {
                'allele_type': 'wild_type',
                'gene': 'Homo sapiens Q00000',
                'name': None,
                'gene_name': 'ABC',
                'primary_identifier': "Q00000:0123456789abcef-1",
                'synonyms': [],
            }
        }
    }
    actual_names = {
        'delta_start': 'ABCdelta',
        'delta_mid': 'ABCdeltaK',
        'delta_end': 'deltaABC',
    }
    # Delta is only replaced at the end of allele names to maintain parity
    # with Canto's behavior, but this might not be what we want.
    expected_names = {
        'delta_start': 'ABC\N{GREEK CAPITAL LETTER DELTA}',
        'delta_mid': 'ABCdeltaK',
        'delta_end': 'deltaABC',
    }

    actual = copy.deepcopy(template)
    expected = copy.deepcopy(template)
    actual_allele = actual['alleles']['Q00000:0123456789abcef-1']
    expected_allele = expected['alleles']['Q00000:0123456789abcef-1']

    for k in actual_names:
        actual_allele['name'] = actual_names[k]
        expected_allele['name'] = expected_names[k]
        add_delta_symbol(actual)
        assert actual == expected


def test_remove_unapproved_sessions():
    actual = {
        'curation_sessions': {
            '0123456789abcdef': {
                'metadata': {
                    'annotation_status': 'APPROVED',
                }
            },
            '1123456789abcdef': {
                'metadata': {
                    'annotation_status': 'CURATION_IN_PROGRESS',
                }
            },
            '2123456789abcdef': {
                'metadata': {
                    'annotation_status': 'APPROVAL_IN_PROGRESS',
                }
            },
        }
    }
    expected = {
        'curation_sessions': {
            '0123456789abcdef': {
                'metadata': {
                    'annotation_status': 'APPROVED',
                }
            }
        }
    }
    remove_unapproved_sessions(actual)
    assert actual == expected


def test_add_chemical_extensions(chemical_data):
    export = {
        'curation_sessions': {
            '1234567890abcdef': {
                'annotations': [
                    {
                        'term': 'PHIPO:0000707',
                        'extension': [],
                    },
                    {
                        'term': 'PHIPO:0000545',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            },
                        ],
                    },
                    {
                        'term': 'PHIPO:0000690',
                        'extension': [],
                    },
                ]
            },
            '1234567890defabc': {
                'annotations': [
                    {
                        'term': 'PHIPO:0000705',
                        'extension': [],
                    },
                    {
                        'term': 'PHIPO:0000708',
                        'extension': [],
                    },
                    {
                        'evidence code': 'Unknown',
                    },
                ]
            },
        }
    }
    expected = {
        'curation_sessions': {
            '1234567890abcdef': {
                'annotations': [
                    {
                        'term': 'PHIPO:0000707',
                        'extension': [
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': 'CHEBI:5827',
                                'relation': 'chebi_id',
                            },
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': '32',
                                'relation': 'frac_code',
                            },
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': '10004-44-1',
                                'relation': 'cas_number',
                            },
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': 'Cc1cc(O)no1',
                                'relation': 'smiles',
                            },
                        ],
                    },
                    {
                        'term': 'PHIPO:0000545',
                        'extension': [
                            {
                                'rangeDisplayName': 'leaf',
                                'rangeType': 'Ontology',
                                'rangeValue': 'BTO:0000713',
                                'relation': 'infects_tissue',
                            },
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': 'CHEBI:3639',
                                'relation': 'chebi_id',
                            },
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': 'M05',
                                'relation': 'frac_code',
                            },
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': 'Clc1c(Cl)c(C#N)c(Cl)c(C#N)c1Cl',
                                'relation': 'smiles',
                            },
                        ],
                    },
                    {
                        'term': 'PHIPO:0000690',
                        'extension': [
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': 'CHEBI:28368',
                                'relation': 'chebi_id',
                            },
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': '303-81-1',
                                'relation': 'cas_number',
                            },
                        ],
                    },
                ]
            },
            '1234567890defabc': {
                'annotations': [
                    {
                        'term': 'PHIPO:0000705',
                        'extension': [
                            {
                                'rangeDisplayName': '',
                                'rangeType': 'Text',
                                'rangeValue': '70458-96-7',
                                'relation': 'cas_number',
                            },
                        ],
                    },
                    {
                        'term': 'PHIPO:0000708',
                        'extension': [],
                    },
                    {
                        'evidence code': 'Unknown',
                    },
                ]
            },
        }
    }
    actual = export
    add_chemical_extensions(actual, chemical_data)
    assert actual == expected


def test_add_uniprot_data_to_genes(export_for_xrefs):
    uniprot_gene_data = {
        'Q00909': {
            'uniprot_id': 'Q00909',
            'name': 'TRI5',
            'product': 'Trichodiene synthase',
            'strain': 'Gibberella zeae (strain ATCC MYA-4620 / CBS 123657 / FGSC 9075 / NRRL 31084 / PH-1)',
            'dbref_gene_id': '23550840',
            'ensembl_sequence_id': [],
            'ensembl_gene_id': [],
        }
    }
    expected = {
        'curation_sessions': {
            '0123456789abcdef': {
                'alleles': {},
                'annotations': [],
                'genes': {
                    'Fusarium graminearum Q00909': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'Q00909',
                        'uniprot_data': {
                            'uniprot_id': 'Q00909',
                            'name': 'TRI5',
                            'product': 'Trichodiene synthase',
                            'strain': 'Gibberella zeae (strain ATCC MYA-4620 / CBS 123657 / FGSC 9075 / NRRL 31084 / PH-1)',
                            'dbref_gene_id': '23550840',
                            'ensembl_sequence_id': [],
                            'ensembl_gene_id': [],
                        },
                    }
                },
                'genotypes': {},
                'metadata': {},
                'metagenotypes': {},
                'organisms': {},
                'publications': {'PMID:12345678': {}},
            }
        },
        'schema_version': 1,
    }
    actual = add_uniprot_data_to_genes(export_for_xrefs, uniprot_gene_data)
    assert actual == expected


def test_add_pubmed_data_to_sessions(export_for_xrefs):
    pubmed_data = {
        'PMID:12345678': {
            'year': '2007',
            'journal_abbr': 'Science',
            'volume': '317',
            'issue': '5843',
            'pages': '1400-2',
            'author': 'Carberry',
        }
    }
    expected = {
        'curation_sessions': {
            '0123456789abcdef': {
                'alleles': {},
                'annotations': [],
                'genes': {
                    'Fusarium graminearum Q00909': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'Q00909',
                    }
                },
                'genotypes': {},
                'metadata': {},
                'metagenotypes': {},
                'organisms': {},
                'publications': {
                    'PMID:12345678': {
                        'pubmed_data': {
                            'year': '2007',
                            'journal_abbr': 'Science',
                            'volume': '317',
                            'issue': '5843',
                            'pages': '1400-2',
                            'author': 'Carberry',
                        }
                    }
                },
            }
        },
        'schema_version': 1,
    }
    actual = add_pubmed_data_to_sessions(export_for_xrefs, pubmed_data)
    assert actual == expected


def test_add_proteome_strains_to_genes(export_with_uniprot_data, proteome_results):
    proteome_id_mapping = {'Q00909': ['UP000070720']}
    expected = {
        'curation_sessions': {
            '0123456789abcdef': {
                'alleles': {},
                'annotations': [],
                'genes': {
                    'Fusarium graminearum Q00909': {
                        'organism': 'Fusarium graminearum',
                        'uniquename': 'Q00909',
                        'uniprot_data': {
                            'dbref_gene_id': '23550840',
                            'ensembl_gene_id': [],
                            'ensembl_sequence_id': [],
                            'name': 'TRI5',
                            'product': 'Trichodiene synthase',
                            'strain': 'PH-1',
                            'uniprot_id': 'Q00909',
                        },
                    }
                },
                'genotypes': {},
                'metadata': {},
                'metagenotypes': {},
                'organisms': {},
                'publications': {'PMID:12345678': {}},
            }
        },
        'schema_version': 1,
    }
    actual = add_proteome_strains_to_genes(
        export_with_uniprot_data, proteome_results, proteome_id_mapping
    )
    assert actual == expected


def test_get_all_uniprot_ids_in_export(export_for_xrefs):
    expected = {'Q00909', 'Q4QGX0'}
    export_for_xrefs = copy.deepcopy(export_for_xrefs)
    genes = export_for_xrefs['curation_sessions']['0123456789abcdef']['genes']
    genes['Leishmania major Q4QGX0'] = {
        'organism': 'Leishmania major',
        'uniquename': 'Q4QGX0',
    }
    actual = get_all_uniprot_ids_in_export(export_for_xrefs)
    assert actual == expected


def test_get_all_pmids_in_export(export_for_xrefs):
    expected = {'12345678', '1'}
    export_for_xrefs = copy.deepcopy(export_for_xrefs)
    publications = export_for_xrefs['curation_sessions']['0123456789abcdef']['publications']
    publications['PMID:1'] = {}
    actual = get_all_pmids_in_export(export_for_xrefs)
    assert actual == expected


def test_truncate_phi4_ids():
    phi4_ids_below_limit = ['PHI:1', 'PHI:2']
    phi4_ids_above_limit = [f'PHI:{n}' for n in range(1, 50)]
    phi4_ids_truncated = [f'PHI:{n}' for n in range(1, 38)]
    phi4_ids_at_limit = ['PHI:100' for n in range(1, 33)]  # length = 255
    export = {
        'curation_sessions': {
            '0123456789abcdef': {
                'annotations': [
                    # annotations with no PHI-base 4 IDs should be unchanged
                    {},
                    # lists below the truncation limit should be unchanged
                    {'phi4_id': phi4_ids_below_limit},
                    # lists above the truncation limit should be truncated
                    {'phi4_id': phi4_ids_above_limit},
                    # lists exactly at the truncation limit should NOT be truncated
                    {'phi4_id': phi4_ids_at_limit},
                ]
            }
        }
    }
    expected = {
        'curation_sessions': {
            '0123456789abcdef': {
                'annotations': [
                    {},
                    {'phi4_id': ['PHI:1', 'PHI:2']},
                    {'phi4_id': phi4_ids_truncated},
                    {'phi4_id': phi4_ids_at_limit},
                ]
            }
        }
    }
    truncate_phi4_ids(export)
    actual = export
    assert actual == expected
