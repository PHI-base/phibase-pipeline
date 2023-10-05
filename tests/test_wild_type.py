import pytest

from phibase_pipeline.wild_type import (
    convert_allele_to_wild_type,
    convert_genotype_to_wild_type,
    convert_metagenotype_to_wild_type,
    get_feature_counts,
    get_wt_feature_mapping,
    get_wt_features,
    is_genotype_wild_type,
    is_metagenotype_wild_type,
)


@pytest.fixture
def session():
    return {
        'alleles': {
            # mutant pathogen allele
            'Q00000:0123456789abcdef-1': {
                'allele_type': 'deletion',
                'gene': 'Escherichia coli Q00000',
                'gene_name': 'ABC',
                'name': 'ABCdelta',
                'primary_identifier': 'Q00000:0123456789abcdef-1',
                'synonyms': [],
            },
            # wild type pathogen allele
            'Q00000:0123456789abcdef-2': {
                'allele_type': 'wild_type',
                'gene': 'Escherichia coli Q00000',
                'gene_name': 'ABC',
                'name': 'ABC+',
                'primary_identifier': 'Q00000:0123456789abcdef-2',
                'synonyms': [],
            },
            # mutant host allele
            'P00001:0123456789abcdef-1': {
                'allele_type': 'deletion',
                'gene': 'Homo sapiens P00001',
                'gene_name': 'CYC',
                'name': 'CYCdelta',
                'primary_identifier': 'P00001:0123456789abcdef-1',
                'synonyms': [],
            },
        },
        'genotypes': {
            # mutant pathogen genotype
            '0123456789abcdef-genotype-1': {
                'loci': [
                    [
                        {
                            'id': 'Q00000:0123456789abcdef-1',
                        }
                    ]
                ],
                'organism_strain': 'Unknown strain',
                'organism_taxonid': 562,
            },
            # wild type pathogen genotype
            '0123456789abcdef-genotype-2': {
                'loci': [
                    [
                        {
                            'id': 'Q00000:0123456789abcdef-2',
                        }
                    ]
                ],
                'organism_strain': 'Unknown strain',
                'organism_taxonid': 562,
            },
            # wild type host genotype
            'Homo-sapiens-wild-type-genotype-Unknown-strain': {
                'loci': [],
                'organism_strain': 'Unknown strain',
                'organism_taxonid': 9606,
            },
        },
        'metagenotypes': {
            # mutant pathogen, wild type host
            '0123456789abcdef-metagenotype-1': {
                'host_genotype': 'Homo-sapiens-wild-type-genotype-Unknown-strain',
                'pathogen_genotype': '0123456789abcdef-genotype-1',
                'type': 'pathogen-host',
            },
            # wild type pathogen, wild type host
            '0123456789abcdef-metagenotype-2': {
                'host_genotype': 'Homo-sapiens-wild-type-genotype-Unknown-strain',
                'pathogen_genotype': '0123456789abcdef-genotype-2',
                'type': 'pathogen-host',
            },
        },
    }


def test_convert_allele_to_wild_type():
    allele_id = 'Q4QGX0:00014f904fb22f27-1'
    allele = {
        'allele_type': 'deletion',
        'gene': 'Leishmania major Q4QGX0',
        'gene_name': 'C14DM',
        'name': 'C14DMdelta',
        'primary_identifier': 'Q4QGX0:00014f904fb22f27-1',
        'synonyms': [],
    }
    expected = {
        'allele_type': 'wild type',
        'gene': 'Leishmania major Q4QGX0',
        'gene_name': 'C14DM',
        'name': 'C14DM+',
        'primary_identifier': 'Q4QGX0:00014f904fb22f27-1',
        'synonyms': [],
    }
    actual = convert_allele_to_wild_type(allele_id, allele)
    assert actual == expected


def test_is_genotype_wild_type():
    session = {
        'alleles': {
            'Q00000:0123456789abcdef-1': {
                'allele_type': 'wild_type',
            },
            'Q00000:0123456789abcdef-2': {
                'allele_type': 'deletion',
            },
        }
    }
    wt_genotype = {
        'loci': [
            [
                {
                    'expression': 'Not assayed',
                    'id': 'Q00000:0123456789abcdef-1',
                }
            ]
        ]
    }
    mutant_genotype = {
        'loci': [
            [
                {
                    'expression': 'Not assayed',
                    'id': 'Q00000:0123456789abcdef-2',
                }
            ]
        ]
    }
    assert is_genotype_wild_type(session, wt_genotype)
    assert not is_genotype_wild_type(session, mutant_genotype)


def test_is_metagenotype_wild_type():
    session = {
        'alleles': {
            'Q00000:0123456789abcdef-1': {
                'allele_type': 'wild_type',
            },
            'Q00000:0123456789abcdef-2': {
                'allele_type': 'deletion',
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
                ]
            },
            '0123456789abcdef-genotype-2': {
                'loci': [
                    [
                        {
                            'expression': 'Not assayed',
                            'id': 'Q00000:0123456789abcdef-2',
                        }
                    ]
                ]
            },
        },
    }
    wt_metagenotype = {'pathogen_genotype': '0123456789abcdef-genotype-1'}
    mutant_metagenotype = {'pathogen_genotype': '0123456789abcdef-genotype-2'}
    assert is_metagenotype_wild_type(session, wt_metagenotype)
    assert not is_metagenotype_wild_type(session, mutant_metagenotype)


def test_get_feature_counts():
    session = {
        'alleles': {
            'Q00000:0123456789abcdef-1': None,
            'Q00000:0123456789abcdef-2': None,
            'P00000:0123456789abcdef-1': None,
        },
        'genotypes': {
            '0123456789abcdef-genotype-1': None,
            # wild type genotypes should not be counted
            'Lolium-perenne-wild-type-genotype-Unknown-strain': None,
        },
        'metagenotypes': {
            '0123456789abcdef-metagenotype-1': None,
            '0123456789abcdef-metagenotype-2': None,
            '0123456789abcdef-metagenotype-3': None,
        },
    }
    feature_counts = {
        'alleles': {
            'Q00000': 2,
            'P00000': 1,
        },
        'genotypes': 1,
        'metagenotypes': 3,
    }
    assert feature_counts == get_feature_counts(session)

    # Test empty feature objects
    session['metagenotypes'] = {}
    feature_counts['metagenotypes'] = 0
    assert feature_counts == get_feature_counts(session)

    session['alleles'] = {}
    feature_counts['alleles'] = 0
    assert feature_counts == get_feature_counts(session)


def test_get_wt_feature_mapping(session):
    feature_counts = {
        'alleles': {
            'Q00000': 2,
            'P00001': 1,
        },
        'genotypes': 2,
        'metagenotypes': 2,
    }
    expected = {
        'alleles': {
            'Q00000:0123456789abcdef-1': 'Q00000:0123456789abcdef-3',
            'P00001:0123456789abcdef-1': 'P00001:0123456789abcdef-2',
        },
        'genotypes': {
            '0123456789abcdef-genotype-1': '0123456789abcdef-genotype-3',
        },
        'metagenotypes': {
            '0123456789abcdef-metagenotype-1': '0123456789abcdef-metagenotype-3',
        },
    }
    actual = get_wt_feature_mapping(feature_counts, session)
    assert actual == expected


def test_convert_genotype_to_wild_type():
    allele_mapping = {'Q00000:0123456789abcdef-1': 'Q00000:0123456789abcdef-2'}
    # Genotype with expression
    no_expr_genotype = {
        'loci': [
            [
                {
                    'expression': 'Overexpression',
                    'id': 'Q00000:0123456789abcdef-1',
                }
            ]
        ],
        'organism_strain': 'Unknown strain',
        'organism_taxonid': 562,
    }
    # Genotype without expression
    expr_genotype = {
        'loci': [
            [
                {
                    'id': 'Q00000:0123456789abcdef-1',
                }
            ]
        ],
        'organism_strain': 'Unknown strain',
        'organism_taxonid': 562,
    }
    expected = {
        'loci': [
            [
                {
                    'expression': 'Not assayed',
                    'id': 'Q00000:0123456789abcdef-2',
                }
            ]
        ],
        'organism_strain': 'Unknown strain',
        'organism_taxonid': 562,
    }
    actual1 = convert_genotype_to_wild_type(allele_mapping, no_expr_genotype)
    assert actual1 == expected
    actual2 = convert_genotype_to_wild_type(allele_mapping, expr_genotype)
    assert actual2 == expected


def test_convert_metagenotype_to_wild_type():
    genotype_mapping = {'0123456789abcdef-genotype-1': '0123456789abcdef-genotype-2'}
    metagenotype = {
        'host_genotype': 'Homo-sapiens-wild-type-genotype-Unknown-strain',
        'pathogen_genotype': '0123456789abcdef-genotype-1',
        'type': 'pathogen-host',
    }
    expected = {
        'host_genotype': 'Homo-sapiens-wild-type-genotype-Unknown-strain',
        'pathogen_genotype': '0123456789abcdef-genotype-2',
        'type': 'pathogen-host',
    }
    actual = convert_metagenotype_to_wild_type(genotype_mapping, metagenotype)
    assert actual == expected


def test_get_wt_features(session):
    feature_mapping = {
        'alleles': {
            'Q00000:0123456789abcdef-1': 'Q00000:0123456789abcdef-3',
            'P00001:0123456789abcdef-1': 'P00001:0123456789abcdef-2',
        },
        'genotypes': {
            '0123456789abcdef-genotype-1': '0123456789abcdef-genotype-3',
        },
        'metagenotypes': {
            '0123456789abcdef-metagenotype-1': '0123456789abcdef-metagenotype-3',
        },
    }
    expected = {
        'alleles': {
            'P00001:0123456789abcdef-2': {
                'allele_type': 'wild type',
                'gene': 'Homo sapiens P00001',
                'gene_name': 'CYC',
                'name': 'CYC+',
                'primary_identifier': 'P00001:0123456789abcdef-2',
                'synonyms': [],
            },
            'Q00000:0123456789abcdef-3': {
                'allele_type': 'wild type',
                'gene': 'Escherichia coli Q00000',
                'gene_name': 'ABC',
                'name': 'ABC+',
                'primary_identifier': 'Q00000:0123456789abcdef-3',
                'synonyms': [],
            },
        },
        'genotypes': {
            '0123456789abcdef-genotype-3': {
                'loci': [
                    [
                        {
                            'expression': 'Not assayed',
                            'id': 'Q00000:0123456789abcdef-3',
                        }
                    ]
                ],
                'organism_strain': 'Unknown strain',
                'organism_taxonid': 562,
            }
        },
        'metagenotypes': {
            '0123456789abcdef-metagenotype-3': {
                'host_genotype': 'Homo-sapiens-wild-type-genotype-Unknown-strain',
                'pathogen_genotype': '0123456789abcdef-genotype-3',
                'type': 'pathogen-host',
            }
        },
    }
    actual = get_wt_features(feature_mapping, session)
    assert actual == expected
