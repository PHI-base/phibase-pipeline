import pandas as pd
from pandas.testing import assert_frame_equal

from phibase_pipeline import spreadsheet


def test_get_entry_summary_table():
    export = {
        'curation_sessions': {
            '0123456879abcdef': {
                'genes': {
                    'Candida albicans O59924': {
                        'uniquename': 'O59924',
                        'organism': 'Candida albicans',
                        'uniprot_data': {
                            'uniprot_id': 'O59924',
                            'name': 'SOD1',
                            'product': 'Superoxide dismutase [Cu-Zn]',
                            'strain': '',
                            'ensembl_sequence_id': ['C4_02320C_A-T-p1'],
                            'dbref_gene_id': '',
                        },
                        'phig_id': 'PHIG:1',
                    }
                },
                'metagenotypes': {
                    '0123456879abcdef-metagenotype-1': {
                        'pathogen_genotype': '0123456879abcdef-genotype-1',
                        'host_genotype': 'Triticum-aestivum-wild-type-genotype-cv.-Svevo',
                        'type': 'pathogen-host',
                    }
                },
                'alleles': {
                    '0123456879abcdef-allele-1': {
                        'gene': 'Candida albicans O59924',
                    },
                },
                'genotypes': {
                    '0123456879abcdef-genotype-1': {
                        'loci': [
                            [
                                {
                                    'id': '0123456879abcdef-allele-1',
                                }
                            ]
                        ]
                    },
                    'Triticum-aestivum-wild-type-genotype-cv.-Svevo': {
                        'loci': [],
                    }
                },
                'annotations': [
                    {
                        'high_level_terms': ['Reduced virulence'],
                        'gene': 'Candida albicans O59924',
                        'phi4_id': ['PHI:4'],
                    },
                    {
                        'high_level_terms': ['Reduced virulence'],
                        'metagenotype': '0123456879abcdef-metagenotype-1',
                        'phi4_id': ['PHI:5'],
                    },
                ],
            }
        }
    }
    expected = pd.DataFrame.from_records(
        [
            {
                'phig_id': 'PHIG:1',
                'gene_name': 'SOD1',
                'species': 'Candida albicans',
                'uniprot_id': 'O59924',
                'gene_product': 'Superoxide dismutase [Cu-Zn]',
                'sequence_strain': '',
                'ensembl_genomes': 'C4_02320C_A-T-p1',
                'ncbi_genbank': '',
                'high_level_term': 'Reduced virulence',
                'phi4_id': 'PHI:4; PHI:5',
            }
        ]
    )
    actual = spreadsheet.get_entry_summary_table(export)
    assert_frame_equal(actual, expected)
