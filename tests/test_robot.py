# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

from phibase_pipeline.robot import get_ontology_terms_and_labels


TEST_DATA_DIR = Path(__file__).parent / 'data'
ROBOT_DATA_DIR = TEST_DATA_DIR / 'robot'


def test_get_ontology_terms_and_labels():
    expected = {
        'TEST:0000001': 'term 1',
        'TEST:0000002': 'term 2',
        'TEST2:0000001': 'term 3',
    }
    term_label_mapping_config = {
        'TEST': ROBOT_DATA_DIR / 'test_ontology_1.owl',
        'TEST2': ROBOT_DATA_DIR / 'test_ontology_2.owl',
    }
    actual = get_ontology_terms_and_labels(term_label_mapping_config)
    assert actual == expected
