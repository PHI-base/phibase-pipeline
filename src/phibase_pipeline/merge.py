import json


def update_session_ids(recurated_sessions):

    def rename_session_ids(old_session_id, new_session_id, session):
        # TODO: Do this replacement properly
        if not session:
            return {}
        return json.loads(
            json.dumps(session).replace(old_session_id, new_session_id)
        )

    renamed_sessions = {}
    old_session_id = None
    new_session_id = None
    for pmid, session_dict in recurated_sessions.items():
        phibase_session = session_dict['phibase']
        if phibase_session:
            old_session_id = phibase_session['metadata']['canto_session']
        canto_session = session_dict['canto']
        if phibase_session:
            new_session_id = canto_session['metadata']['canto_session']
        renamed_sessions[pmid] = {
            'phibase': rename_session_ids(
                old_session_id, new_session_id, phibase_session
            ),
            'canto': canto_session
        }
    return renamed_sessions


def get_recurated_sessions(phibase_export, canto_export):
    phibase_pmid_sessions, canto_pmid_sessions = (
        {
            session['metadata']['curation_pub_id']: session
            for session in export['curation_sessions'].values()
        }
        for export in (phibase_export, canto_export)
    )
    recurated_pmids = set.intersection(
        set(phibase_pmid_sessions.keys()),
        set(canto_pmid_sessions.keys()),
    )
    recurated_sessions = {
        pmid: {
            'phibase': phibase_pmid_sessions[pmid],
            'canto': canto_pmid_sessions[pmid],
        }
        for pmid in recurated_pmids
    }
    return recurated_sessions


def rekey_duplicate_feature_ids(feature_type, phibase_session, canto_session):

    def get_feature_number(feature_id):
        num = feature_id.rsplit('-')[-1]
        return int(num) if num.isdigit() else 0

    def get_max_feature_number(features):
        return max(
            get_feature_number(feature_id)
            for feature_id in features.keys()
        )

    def get_matching_feature_ids(features_a, features_b):
        ids_a = features_a.keys()
        ids_b = list(features_b.keys())
        # maintain original order to ensure consistent rekeying later
        return sorted(
            set(ids_a).intersection(ids_b),
            key=lambda k: ids_b.index(k)
        )

    def features_are_unique(feature_type, feature_a, feature_b):
        match feature_type:
            case 'alleles':
                return (
                    feature_a['allele_type'] != feature_b['allele_type']
                    or feature_a['gene'] != feature_b['gene']
                    or feature_a['primary_identifier'] != feature_b['primary_identifier']
                )
            case 'genotypes':
                return (
                    feature_a['loci'] != feature_b['loci']
                    or feature_a['organism_strain'] != feature_b['organism_strain']
                    or feature_a['organism_taxonid'] != feature_b['organism_taxonid']
                )
            case 'metagenotypes':
                return (
                    feature_a['pathogen_genotype'] != feature_b['pathogen_genotype']
                    or feature_a['host_genotype'] != feature_b['host_genotype']
                )
            case _:
                raise ValueError(f'unsupported feature type: {feature_type}')

    if feature_type not in canto_session:
        return {}

    exports = (phibase_session, canto_session)
    features = tuple(e[feature_type] for e in exports)
    next_feature_num = 1 + max(map(get_max_feature_number, features))
    features_a = phibase_session[feature_type]
    features_b = canto_session[feature_type]
    matching_ids = get_matching_feature_ids(features_a, features_b)
    rekeyed_features = features_b.copy()
    for feature_id in matching_ids:
        if 'wild-type' in feature_id:
            continue
        feature_a = features_a[feature_id]
        feature_b = features_b[feature_id]
        if features_are_unique(feature_type, feature_a, feature_b):
            feature_no_num = feature_id[:feature_id.rindex('-')]
            new_feature_id = f'{feature_no_num}-{next_feature_num}'
            rekeyed_features[new_feature_id] = feature_b
            del rekeyed_features[feature_id]
            next_feature_num += 1

    return rekeyed_features


def merge_recurated_sessions(recurated_sessions):
    merged_sessions = {}
    for pmid, session_dict in recurated_sessions.items():
        phibase_session, canto_session = session_dict['phibase'], session_dict['canto']
        if 'genes' not in canto_session:
            continue  # session is not approved or not valid
        merged_session = phibase_session.copy()
        for feature_type in ('genes', 'organisms'):
            merged_session[feature_type].update(canto_session[feature_type])
        for feature_type in ('alleles', 'genotypes', 'metagenotypes'):
            rekeyed_features = rekey_duplicate_feature_ids(
                feature_type, phibase_session, canto_session
            )
            merged_session[feature_type].update(rekeyed_features)
        merged_session['annotations'].extend(canto_session['annotations'])
        merged_session['metadata'] = canto_session['metadata']
        merged_session['metadata']['session_genes_count'] = len(merged_session['genes'])
        merged_sessions[pmid] = merged_session

    return merged_sessions
