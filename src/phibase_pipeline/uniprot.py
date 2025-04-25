# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""
Retrieve data from the UniProtKB ID mapping service. Adapted from the
Python example on the UniProt ID mapping help page:
https://www.uniprot.org/help/id_mapping
"""

import json
import re
import time
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode

import requests
from requests.adapters import HTTPAdapter, Retry


def make_session():
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise


def submit_id_mapping(from_db, to_db, ids):
    url = 'https://rest.uniprot.org/idmapping/run'
    ids = ','.join(ids)
    data = {
        'from': from_db,
        'to': to_db,
        'ids': ids,
    }
    request = requests.post(url, data)
    check_response(request)
    return request.json()['jobId']


def check_id_mapping_results_ready(session, job_id, poll_seconds=3):
    url = f'https://rest.uniprot.org/idmapping/status/{job_id}'
    while True:
        request = session.get(url)
        check_response(request)
        json_data = request.json()
        if 'jobStatus' in json_data:
            if json_data['jobStatus'] == 'RUNNING':
                time.sleep(poll_seconds)
            else:
                raise Exception(json_data['jobStatus'])
        else:
            return bool(json_data['results'] or json_data['failedIds'])


def get_id_mapping_results_link(session, job_id):
    url = f'https://rest.uniprot.org/idmapping/details/{job_id}'
    request = session.get(url)
    check_response(request)
    return request.json()['redirectURL']


def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        match file_format:
            case 'json':
                json_data = json.loads(decompressed.decode('utf-8'))
                return json_data
            case 'tsv':
                lines = decompressed.decode('utf-8').split('\n')
                return [line for line in lines if line]
            case 'xlsx':
                return [decompressed]
            case 'xml':
                return [decompressed.decode('utf-8')]
            case _:
                return decompressed.decode('utf-8')
    else:
        match file_format:
            case 'json':
                return response.json()
            case 'tsv':
                return [line for line in response.text.split('\n') if line]
            case 'xlsx':
                return [response.content]
            case 'xml':
                return [response.text]
            case _:
                pass
    return response.text


def get_batch(session, batch_response, file_format, compressed):

    def get_next_link(headers):
        re_next_link = re.compile(r"<(.+)>; rel='next'")
        if 'Link' in headers:
            match = re_next_link.match(headers['Link'])
            if match:
                return match.group(1)

    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)


def combine_batches(all_results, batch_results, file_format):
    if file_format == 'json':
        for key in ('results', 'failedIds'):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == 'tsv':
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_search(session, url):

    def merge_xml_results(xml_results):

        def get_xml_namespace(element):
            m = re.match(r'\{(.*)\}', element.tag)
            return m.groups()[0] if m else ''

        merged_root = ElementTree.fromstring(xml_results[0])
        for result in xml_results[1:]:
            root = ElementTree.fromstring(result)
            for child in root.findall(r'{http://uniprot.org/uniprot}entry'):
                merged_root.insert(-1, child)
        ElementTree.register_namespace('', get_xml_namespace(merged_root[0]))
        return ElementTree.tostring(merged_root, encoding='utf-8', xml_declaration=True)

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query['format'][0] if 'format' in query else 'json'
    if 'size' in query:
        size = int(query['size'][0])
    else:
        size = 500
        query['size'] = size
    compressed = (
        query['compressed'][0].lower() == 'true' if 'compressed' in query else False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))
    url = parsed.geturl()
    request = session.get(url)
    check_response(request)
    results = decode_results(request, file_format, compressed)
    for batch in get_batch(session, request, file_format, compressed):
        results = combine_batches(results, batch, file_format)
    if file_format == 'xml':
        results = merge_xml_results(results)
    return results


def retrieve_id_mapping(session, from_db, to_db, ids, poll_seconds=3):
    job_id = submit_id_mapping(from_db, to_db, ids)
    if check_id_mapping_results_ready(session, job_id, poll_seconds):
        link = get_id_mapping_results_link(session, job_id)
        results = get_id_mapping_results_search(session, link)
        return results


def run_id_mapping_job(session, ids):
    results = retrieve_id_mapping(
        session,
        from_db='UniProtKB_AC-ID',
        to_db='UniProtKB',
        ids=ids,
        poll_seconds=30,
    )
    return results


def get_uniprot_data_fields(id_mapping_results):
    all_gene_data = {}
    all_results = id_mapping_results['results']
    for result in all_results:
        gene_data = {
            'uniprot_id': None,
            'name': None,
            'product': None,
            'strain': None,
            'dbref_gene_id': None,
            'ensembl_sequence_id': None,
            'ensembl_gene_id': None,
        }
        uniprot_id = result['from']
        data = result['to']

        # Primary UniProt ID (in case of merged accessions)
        gene_data['uniprot_id'] = data['primaryAccession']

        if data['entryType'] == 'Inactive':
            # There's nothing more we can add for inactive accessions
            all_gene_data[uniprot_id] = gene_data
            continue

        # Gene name
        genes = data.get('genes')
        if genes:
            # TODO: Add all gene names (but should we join here or not?)
            gene = genes[0]
            if 'geneName' in gene:
                gene_data['name'] = gene['geneName']['value']
            elif 'orfNames' in gene:
                # TODO: Add all ORF names (but should we join here or not?)
                gene_data['name'] = gene['orfNames'][0]['value']

        # Gene product
        protein = data['proteinDescription']
        for name_key in ('recommendedName', 'submittedName', 'submissionNames'):
            protein_name_data = protein.get(name_key)
            if protein_name_data:
                if name_key == 'submissionNames':
                    # TODO: Decide which submission name to use
                    protein_name_data = protein_name_data[0]
                gene_data['product'] = protein_name_data['fullName']['value']
                break

        # Strain
        gene_data['strain'] = data['organism']['scientificName']

        # Database reference gene ID
        dbrefs = data['uniProtKBCrossReferences']
        dbref_ids = [
            dbref['id'] for dbref in dbrefs if dbref['database'] == 'GeneID'
        ]
        # TODO: Decide whether to display multiple gene IDs
        gene_id = dbref_ids[0] if dbref_ids else None
        gene_data['dbref_gene_id'] = gene_id

        # Ensembl sequence ID
        ensembl_refs = [dbref for dbref in dbrefs if dbref['database'].startswith('Ensembl')]
        ensembl_protein_ids = [
            prop['value']
            for dbref in ensembl_refs
            for prop in dbref['properties']
            if prop['key'] == 'ProteinId'
        ]
        gene_data['ensembl_sequence_id'] = ensembl_protein_ids

        # Ensembl gene ID
        ensembl_gene_ids = [
            prop['value']
            for dbref in ensembl_refs
            for prop in dbref['properties']
            if prop['key'] == 'GeneId'
        ]
        gene_data['ensembl_gene_id'] = ensembl_gene_ids
        all_gene_data[uniprot_id] = gene_data

    return all_gene_data


def get_proteome_id_mapping(id_mapping_results):
    proteome_id_mapping = {}
    for result in id_mapping_results:
        original_id = result['from']
        data = result['to']
        if data['entryType'] == 'Inactive':
            continue
        proteome_ids = [
            xref['id'] for xref in data['uniProtKBCrossReferences']
            if xref['database'] == 'Proteomes'
        ]
        proteome_id_mapping[original_id] = proteome_ids
    return proteome_id_mapping


def query_proteome_ids(session, proteome_id_mapping):
    proteome_ids = list(set(
        pid
        for pids in proteome_id_mapping.values()
        for pid in pids
    ))
    # Chunk the proteome IDs so to not exceed the query limit
    chunk_size = 500
    id_chunks = []
    for i in range(0, len(proteome_ids), chunk_size):
        id_chunks.append(proteome_ids[i : i + chunk_size])

    url = 'https://rest.uniprot.org/proteomes/search'
    params = {
        'format': 'json',
        'size': '500',
        'compressed': 'true',
    }
    request = requests.PreparedRequest()

    all_results = []
    for proteome_ids in id_chunks:
        query = f"({'+OR+'.join(f'(upid:{pid})' for pid in proteome_ids)})"
        params['query'] = query
        request.prepare_url(url, params)
        # UniProt seems to need the plus symbol to be unescaped?
        batch_url = request.url.replace('%2B', '+')
        results = get_id_mapping_results_search(session, batch_url)
        all_results.append(results)

    combined_results = {'results': []}
    for results in all_results:
        combined_results = combine_batches(combined_results, results, file_format=params['format'])
    return combined_results
