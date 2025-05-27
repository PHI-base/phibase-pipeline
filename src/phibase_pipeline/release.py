# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT


import json

from phibase_pipeline import ensembl, migrate, postprocess, validate


def release_to_zenodo(args):
    canto_json = migrate.make_combined_export(args.phibase, args.phicanto)
    canto_json = postprocess.add_cross_references(canto_json)
    validate.validate_export(canto_json)
    with open(args.output, 'w+', encoding='utf8') as json_file:
        json.dump(canto_json, json_file, indent=4, sort_keys=True, ensure_ascii=False)


def release_to_ensembl(args):
    ensembl.write_ensembl_exports(
        phibase_path=args.phibase,
        canto_export_path=args.phicanto,
        uniprot_data_path=args.uniprot_data,
        dir_path=args.out_dir,
    )
