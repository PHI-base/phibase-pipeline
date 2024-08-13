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
        return set.intersection(
            set(features_a.keys()),
            set(features_b.keys())
        )

    rekeyed_features = {}
    exports = (phibase_session, canto_session)
    features = tuple(e[feature_type] for e in exports)
    next_feature_num = 1 + max(map(get_max_feature_number, features))
    features_a = phibase_session[feature_type]
    features_b = canto_session[feature_type]
    matching_ids = get_matching_feature_ids(features_a, features_b)
    for feature_id in matching_ids:
        feature_a = features_a[feature_id]
        feature_b = features_b[feature_id]
        if (feature_a != feature_b) and 'wild-type' not in feature_id:
            feature_no_num = feature_id[:feature_id.rindex('-')]
            new_feature_id = f'{feature_no_num}-{next_feature_num}'
            rekeyed_features[new_feature_id] = feature_b
            next_feature_num += 1

    return rekeyed_features
