# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""Command line interface for the phibase-pipeline package."""

import argparse
import json

from phibase_pipeline import ensembl, migrate, postprocess, validate


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog='phibase_pipeline',
        description='Pipeline for version 5 of the PHI-base database.',
    )
    subparsers = parser.add_subparsers(dest='target', required=True)
    parser_zenodo = subparsers.add_parser('zenodo')
    parser_ensembl = subparsers.add_parser('ensembl')

    shared_args = {
        'phibase': {
            'metavar': 'PHIBASE_CSV',
            'type': str,
            'help': 'the path to the PHI-base 4 CSV export file',
        },
        'phicanto': {
            'metavar': 'CANTO_JSON',
            'type': str,
            'help': 'the path to the PHI-Canto JSON export file',
        },
    }
    for subparser in (parser_zenodo, parser_ensembl):
        for arg_name, kwargs in shared_args.items():
            subparser.add_argument(arg_name, **kwargs)

    parser_zenodo.add_argument(
        'output',
        metavar='OUTFILE',
        type=str,
        help='the path to write the combined JSON export file',
    )

    parser_ensembl.add_argument(
        'uniprot_data',
        metavar='UNIPROT_DATA',
        type=str,
        help='the path to the UniProt data file',
    )
    parser_ensembl.add_argument(
        'out_dir',
        metavar='DIR',
        type=str,
        help='the output directory for the Ensembl export files',
    )
    return parser.parse_args(args)


def run(args):
    args = parse_args(args)
    if args.target == 'zenodo':
        canto_json = migrate.make_combined_export(args.phibase, args.phicanto)
        canto_json = postprocess.add_cross_references(canto_json)
        validate.validate_export(canto_json)
        with open(args.output, 'w+', encoding='utf8') as json_file:
            json.dump(canto_json, json_file, indent=4, sort_keys=True, ensure_ascii=False)
    elif args.target == 'ensembl':
        ensembl.write_ensembl_exports(
            phibase_path=args.phibase,
            canto_export_path=args.phicanto,
            uniprot_data_path=args.uniprot_data,
            dir_path=args.out_dir,
        )
    else:
        # argparse should prevent this from being reached
        raise ValueError(f'unsupported target type: {args.target}')
