# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""
Release the PHI-base 5 dataset for loading into the PHI-base 5 database,
or for uploading to the Zenodo repository or Ensembl.
"""

import json

import pandas as pd

from phibase_pipeline import ensembl, loaders, migrate, postprocess, robot, spreadsheet, validate


def get_release_stats_table(export):
    def iter_feature_keys(feature_key):
        for session in export['curation_sessions'].values():
            features = session.get(feature_key, {})
            for feature_id in features:
                yield feature_id

    def iter_taxon_ids(role):
        for session in export['curation_sessions'].values():
            for taxon_id, organism in session['organisms'].items():
                if organism['role'] == role:
                    yield taxon_id

    def count_unique_annotations(annotation_type):
        return len(set(
            annotation['term']
            for session in export['curation_sessions'].values()
            for annotation in session['annotations']
            if annotation['type'] == annotation_type
        ))

    def count_annotations(annotation_type):
        return len(list(
            annotation
            for session in export['curation_sessions'].values()
            for annotation in session['annotations']
            if annotation['type'] == annotation_type
        ))

    count_features = lambda k: len(set(iter_feature_keys(k)))

    annotation_types = {
        'Pathogen-host interaction phenotype': 'pathogen_host_interaction_phenotype',
        'Gene-for-gene phenotype': 'gene_for_gene_phenotype',
        'Pathogen phenotype': 'pathogen_phenotype',
        'Host phenotype': 'host_phenotype',
        'GO biological process': 'biological_process',
        'GO cellular component': 'cellular_component',
        'GO molecular function': 'molecular_function',
        'Post-translational modification': 'post_translational_modification',
        'Physical interaction': 'physical_interaction',
        'WT RNA expression': 'wt_rna_expression',
        'WT protein expression': 'wt_protein_expression',
    }
    annotation_counts = {
        label: count_annotations(a_type) if type(a_type) == str else a_type
        for label, a_type in annotation_types.items()
    }
    data_stats = {
        'Genes': count_features('genes'),
        'Interactions': count_features('metagenotypes'),
        'Pathogen species': len(set(iter_taxon_ids('pathogen'))),
        'Host species': len(set(iter_taxon_ids('host'))),
        'Diseases': count_unique_annotations('disease_name'),
        'References': count_features('publications'),
        **annotation_counts,
    }
    stats_df = (
        pd.DataFrame(
            data_stats,
            columns=list(data_stats.keys()),
            index=['Count']
        )
        .transpose()
        .rename_axis('Data type')
    )
    return stats_df


def write_json_export(path, canto_json):
    with open(path, 'w+', encoding='utf8') as json_file:
        json.dump(canto_json, json_file, indent=4, sort_keys=True, ensure_ascii=False)


def release_to_zenodo(args):
    phi_df = loaders.load_phibase_csv(args.phibase)
    phicanto_json = loaders.load_json(args.phicanto)
    # JSON export format
    canto_json = migrate.make_combined_export(phi_df, phicanto_json)
    canto_json = postprocess.add_cross_references(canto_json)
    validate.validate_export(canto_json)
    write_json_export(args.output_json, canto_json)
    # Spreadsheet export format
    term_label_mapping_config = loaders.load_json(args.term_label_mapping_config)
    term_label_mapping = robot.get_ontology_terms_and_labels(
        term_label_mapping_config
    )
    phig_mapping = loaders.load_phig_uniprot_mapping()
    spreadsheet.make_spreadsheet_from_export(
        export=canto_json,
        phig_mapping=phig_mapping,
        term_label_mapping=term_label_mapping,
        output_path=args.output_xlsx,
    )


def release_to_phibase5(args):
    phi_df = loaders.load_phibase_csv(args.phibase)
    phicanto_json = loaders.load_json(args.phicanto)
    canto_json = migrate.make_combined_export(phi_df, phicanto_json)
    canto_json = postprocess.add_cross_references(canto_json)
    postprocess.truncate_long_values(canto_json)
    validate.validate_export(canto_json, validate_id_length=True)
    write_json_export(args.output, canto_json)


def release_to_ensembl(args):
    ensembl.write_ensembl_exports(
        phibase_path=args.phibase,
        canto_export_path=args.phicanto,
        uniprot_data_path=args.uniprot_data,
        dir_path=args.out_dir,
    )
