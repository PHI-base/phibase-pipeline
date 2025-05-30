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
    for subparser in (parser_zenodo, parser_phibase5, parser_ensembl):
        for arg_name, kwargs in shared_args.items():
            subparser.add_argument(arg_name, **kwargs)

    for subparser in (parser_zenodo, parser_phibase5):
        subparser.add_argument(
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
