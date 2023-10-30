import pytest

from phibase_pipeline.postprocess import (
    allele_ids_of_genotype,
    merge_phi_canto_curation,
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
            }
        }
    }

    merge_phi_canto_curation(phi_base_export, phi_canto_export)
    assert phi_base_export == expected

    # Merging sessions that already exist
    with pytest.raises(KeyError, match='session_id [0-9a-f]{16} already exists'):
        # Merging the same data again will trigger the exception
        merge_phi_canto_curation(phi_base_export, phi_canto_export)
