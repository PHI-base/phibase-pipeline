# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""Command line interface for the phibase-pipeline package."""

import argparse

from phibase_pipeline import release


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog='phibase_pipeline',
        description='Pipeline for version 5 of the PHI-base database.',
    )
    subparsers = parser.add_subparsers(dest='target', required=True)
    parser_zenodo = subparsers.add_parser('zenodo')
    parser_phibase5 = subparsers.add_parser('phibase5')
    parser_ensembl = subparsers.add_parser('ensembl')

    common_args = {
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
    for subparser in subparsers.choices.values():
        for arg_name, kwargs in common_args.items():
            subparser.add_argument(arg_name, **kwargs)

    parser_zenodo.add_argument(
        'term_label_mapping_config',
        metavar='MAPPING_CONFIG',
        type=str,
        help='JSON mapping of ontology prefixes to paths',
    )
    parser_zenodo.add_argument(
        'output_json',
        metavar='JSON_OUTPUT',
        type=str,
        help='the path to write the combined JSON export file',
    )
    parser_zenodo.add_argument(
        'output_xlsx',
        metavar='SPREADSHEET_OUTPUT',
        type=str,
        help='the path to write the spreadsheet export format',
    )
    
    parser_phibase5.add_argument(
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
    match args.target:
        case 'zenodo':
            release.release_to_zenodo(args)
        case 'phibase5':
            release.release_to_phibase5(args)
        case 'ensembl':
            release.release_to_ensembl(args)
        case _:
            # argparse should prevent this from being reached
            raise ValueError(f'unsupported target type: {args.target}')
