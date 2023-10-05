from phibase_pipeline.wild_type import convert_allele_to_wild_type


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
