# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""
Release the PHI-base 5 dataset for loading into the PHI-base 5 database,
or for uploading to the Zenodo repository or Ensembl.
"""

import json

from phibase_pipeline import ensembl, loaders, migrate, postprocess, robot, spreadsheet, validate


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
    spreadsheet.make_spreadsheet_from_export(
        export=canto_json,
        phig_mapping=args.phig_mapping,
        term_label_mapping=term_label_mapping,
        output_path=args.output_xlsx,
    )


def release_to_phibase5(args):
    phi_df = loaders.load_phibase_csv(args.phibase)
    phicanto_json = loaders.load_json(args.phicanto)
    canto_json = migrate.make_combined_export(phi_df, phicanto_json)
    canto_json = postprocess.add_cross_references(canto_json)
    postprocess.truncate_phi4_ids(canto_json)
    validate.validate_export(canto_json, validate_id_length=True)
    write_json_export(args.output, canto_json)


def release_to_ensembl(args):
    ensembl.write_ensembl_exports(
        phibase_path=args.phibase,
        canto_export_path=args.phicanto,
        uniprot_data_path=args.uniprot_data,
        dir_path=args.out_dir,
    )
