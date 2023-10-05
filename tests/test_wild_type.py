from phibase_pipeline.wild_type import (
    convert_allele_to_wild_type,
    is_genotype_wild_type,
    is_metagenotype_wild_type,
    get_feature_counts,
)


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
