# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from phibase_pipeline.pubmed import get_all_publication_details

TEST_DATA_DIR = Path(__file__).parent / 'data'
PUBMED_DATA_DIR = TEST_DATA_DIR / 'pubmed'


@pytest.fixture
def pubmed_response():
    path = PUBMED_DATA_DIR / 'pubmed_response.xml'
    with open(path, encoding='utf-8') as xml_file:
        return xml_file.read()


def test_get_all_publication_details(pubmed_response):
    expected = {
        'PMID:12345678': {
            'year': '2007',
            'journal_abbr': 'Science',
            'volume': '317',
            'issue': '5843',
            'pages': '1400-2',
            'author': 'Carberry',
        }
    }
    actual = get_all_publication_details(pubmed_response)
    assert actual == expected
