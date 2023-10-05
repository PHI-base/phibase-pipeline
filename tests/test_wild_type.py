from phibase_pipeline.wild_type import (
    convert_allele_to_wild_type,
    is_genotype_wild_type,
    is_metagenotype_wild_type,
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
