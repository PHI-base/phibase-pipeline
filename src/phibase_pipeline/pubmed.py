# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""
Get publication details from PubMed.
"""

from xml.etree import ElementTree

import requests


def get_publications_from_pubmed(pmids):
    pmid_str = ','.join(pmids)
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    data = {
        'db': 'pubmed',
        'id': pmid_str,
        'retmode': 'xml',
    }
    response = requests.post(url, data=data)
    pubmed_data = response.content
    return pubmed_data


def get_all_publication_details(pubmed_data):

    def get_author_name(article):
        author = article.find('AuthorList/Author')
        last_name = author.find('LastName')
        if last_name is not None:
            author_name = last_name.text
        else:
            collective_name = author.find('CollectiveName')
            if collective_name is None:
                raise ValueError(f'no author name found for {pmid}')
            author_name = collective_name.text
        return author_name

    maybe_text = lambda e: e.text if e is not None else None

    all_pubmed_data = {}
    root = ElementTree.fromstring(pubmed_data)
    for element in root:
        if element.tag == 'PubmedArticle':
            year = None
            volume = None
            issue = None
            pmid = 'PMID:' + element.find('MedlineCitation/PMID').text
            article = element.find('MedlineCitation/Article')
            journal = article.find('Journal')
            journal_abbr = journal.find('ISOAbbreviation')
            journal_issue = journal.find('JournalIssue')
            if journal_issue is not None:
                year = journal_issue.find('PubDate/Year')
                volume = journal_issue.find('Volume')
                issue = journal_issue.find('Issue')
            pages = article.find('Pagination/MedlinePgn')
            author_name = get_author_name(article)
            all_pubmed_data[pmid] = {
                'year': maybe_text(year),
                'journal_abbr': journal_abbr.text,
                'volume': maybe_text(volume),
                'issue': maybe_text(issue),
                'pages': maybe_text(pages),
                'author': author_name,
            }
    return all_pubmed_data
