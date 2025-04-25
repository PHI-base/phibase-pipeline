# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from phibase_pipeline.loaders import load_json
from phibase_pipeline.uniprot import get_uniprot_data_fields

TEST_DATA_DIR = Path(__file__).parent / 'data'
UNIPROT_DATA_DIR = TEST_DATA_DIR / 'uniprot'


@pytest.mark.parametrize(
    'path,expected',
    (
        pytest.param(
            UNIPROT_DATA_DIR / 'protein_result_P05067.json',
            {
                'P05067': {
                    'uniprot_id': 'P05067',
                    'name': 'APP',
                    'product': 'Amyloid-beta precursor protein',
                    'strain': 'Homo sapiens',
                    'dbref_gene_id': '351',
                    'ensembl_sequence_id': [
                        'ENSP00000284981.4',
                        'ENSP00000345463.5',
                        'ENSP00000346129.3',
                        'ENSP00000350578.3',
                        'ENSP00000351796.3',
                        'ENSP00000387483.2',
                    ],
                    'ensembl_gene_id': [
                        'ENSG00000142192.22',
                        'ENSG00000142192.22',
                        'ENSG00000142192.22',
                        'ENSG00000142192.22',
                        'ENSG00000142192.22',
                        'ENSG00000142192.22',
                    ],
                }
            },
            id='P05067',
        ),
        pytest.param(
            UNIPROT_DATA_DIR / 'protein_result_Q00909.json',
            {
                'Q00909': {
                    'uniprot_id': 'Q00909',
                    'name': 'TRI5',
                    'product': 'Trichodiene synthase',
                    'strain': 'Gibberella zeae (strain ATCC MYA-4620 / CBS 123657 / FGSC 9075 / NRRL 31084 / PH-1)',
                    'dbref_gene_id': '23550840',
                    'ensembl_sequence_id': [],
                    'ensembl_gene_id': [],
                }
            },
            id='Q00909',
        ),
    ),
)
def test_get_uniprot_data_fields(path, expected):
    test_data = load_json(path)
    actual = get_uniprot_data_fields(test_data)
    assert actual == expected
