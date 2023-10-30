from phibase_pipeline.postprocess import (
    allele_ids_of_genotype,
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
