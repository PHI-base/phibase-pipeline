import copy
import numpy as np
import re
from collections import defaultdict


def is_genotype_wild_type(session, genotype):
    for locus in genotype['loci']:
        for genotype_allele in locus:
            allele = session['alleles'][genotype_allele['id']]
            if allele['allele_type'] != 'wild_type':
                return False
    return True  # genotypes with no alleles are wild-type


def is_metagenotype_wild_type(session, metagenotype):
    pathogen_genotype_id = metagenotype['pathogen_genotype']
    pathogen_genotype = session['genotypes'][pathogen_genotype_id]
    return is_genotype_wild_type(session, pathogen_genotype)


def get_feature_counts(session):
    feature_counts = {}
    id_patterns = {
        'alleles': r'^(?P<protein>[A-Z0-9]+):[a-f0-9]{16}-(?P<n>\d+)$',
        'genotypes': r'^[a-f0-9]{16}-genotype-(?P<n>\d+)$',
        'metagenotypes': r'^[a-f0-9]{16}-metagenotype-(?P<n>\d+)$',
    }
    for feature_key in ('genotypes', 'metagenotypes'):
        pattern = id_patterns[feature_key]
        id_numbers = []
        feature_ids = session[feature_key]
        if not feature_ids:
            continue  # session may not have metagenotypes
        for feature_id in feature_ids:
            match = re.match(pattern, feature_id)
            if not match:
                continue  # wild type genotypes should not match
            id_num = int(match.group('n'))
            id_numbers.append(id_num)
        feature_counts[feature_key] = max(id_numbers)

    # Additional indexing is needed for alleles
    allele_numbers = defaultdict(list)
    for allele_id in session['alleles']:
        match = re.match(id_patterns['alleles'], allele_id)
        protein_id = match.group('protein')
        id_num = int(match.group('n'))
        allele_numbers[protein_id].append(id_num)

    feature_counts['alleles'] = {
        protein_id: max(id_nums) for protein_id, id_nums in allele_numbers.items()
    }
    return feature_counts


def get_wt_feature_mapping(feature_counts, session):
    feature_keys = ('alleles', 'genotypes', 'metagenotypes')
    wt_functions = {
        'alleles': lambda _, x: x['allele_type'] == 'wild_type',
        'genotypes': is_genotype_wild_type,
        'metagenotypes': is_metagenotype_wild_type,
    }
    feature_mapping = {k: {} for k in feature_keys}

    for feature_key in feature_keys:
        features = session[feature_key]
        is_wild_type = wt_functions[feature_key]
        for feature_id, feature in features.items():
            if is_wild_type(session, feature):
                continue
            if feature_key == 'alleles':
                protein_id = feature_id.split(':')[0]
                feature_counts[feature_key][protein_id] += 1
                id_str = str(feature_counts[feature_key][protein_id])
            else:
                feature_counts[feature_key] += 1
                id_str = str(feature_counts[feature_key])
            wt_feature_id = re.sub(r'\d+$', id_str, feature_id)
            feature_mapping[feature_key][feature_id] = wt_feature_id
    return feature_mapping


def convert_allele_to_wild_type(allele_id, allele):
    new_allele = allele.copy()
    gene_name = allele['name'] if allele['gene_name'] is np.nan else allele['gene_name']
    new_allele['allele_type'] = 'wild type'
    new_allele['name'] = gene_name + '+'
    new_allele['primary_identifier'] = allele_id
    if 'description' in new_allele:
        del new_allele['description']
    # TODO: Change allele synonyms to be wild type
    new_allele['synonyms'] = []
    return new_allele


def convert_genotype_to_wild_type(allele_mapping, genotype):
    genotype = copy.deepcopy(genotype)
    for locus in genotype['loci']:
        for genotype_allele in locus:
            if 'expression' in genotype_allele:
                genotype_allele['expression'] = 'Not assayed'
            mutant_allele_id = genotype_allele['id']
            genotype_allele['id'] = allele_mapping[mutant_allele_id]
    return genotype


def convert_metagenotype_to_wild_type(genotype_mapping, metagenotype):
    metagenotype = metagenotype.copy()
    pathogen_genotype_id = metagenotype['pathogen_genotype']
    wt_pathogen_genotype_id = genotype_mapping[pathogen_genotype_id]
    metagenotype['pathogen_genotype'] = wt_pathogen_genotype_id
    return metagenotype


def get_wt_features(feature_mapping, session):
    wt_functions = {
        'alleles': {
            'function': convert_allele_to_wild_type,
            'arg': None,
        },
        'genotypes': {
            'function': convert_genotype_to_wild_type,
            'arg': feature_mapping['alleles'],
        },
        'metagenotypes': {
            'function': convert_metagenotype_to_wild_type,
            'arg': feature_mapping['genotypes'],
        },
    }
    wt_features = defaultdict(dict)
    for feature_key, mapping in feature_mapping.items():
        function_data = wt_functions[feature_key]
        conversion_function = function_data['function']
        mapping_arg = function_data['arg']
        for mutant_id, wt_id in mapping.items():
            mutant_feature = session[feature_key][mutant_id]
            if feature_key == 'alleles':
                mapping_arg = wt_id
            wt_feature = conversion_function(mapping_arg, mutant_feature)
            wt_features[feature_key][wt_id] = wt_feature
    return wt_features


def get_all_feature_mappings(canto_json):
    feature_mappings = {}
    curation_sessions = canto_json['curation_sessions']
    for session_id, session in curation_sessions.items():
        feature_counts = get_feature_counts(session)
        feature_mapping = get_wt_feature_mapping(feature_counts, session)
        feature_mappings[session_id] = feature_mapping
    return feature_mappings
