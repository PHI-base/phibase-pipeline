# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""
Release the PHI-base 5 dataset for loading into the PHI-base 5 database,
or for uploading to the Zenodo repository or Ensembl.
"""

import json

from phibase_pipeline import ensembl, migrate, postprocess, validate


def write_json_export(path, canto_json):
    with open(path, 'w+', encoding='utf8') as json_file:
        json.dump(canto_json, json_file, indent=4, sort_keys=True, ensure_ascii=False)


def release_to_zenodo(args):
    canto_json = migrate.make_combined_export(args.phibase, args.phicanto)
    canto_json = postprocess.add_cross_references(canto_json)
    validate.validate_export(canto_json)
    write_json_export(args.output, canto_json)


def release_to_phibase5(args):
    canto_json = migrate.make_combined_export(args.phibase, args.phicanto)
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
