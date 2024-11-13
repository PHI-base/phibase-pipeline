import argparse
import json

from phibase_pipeline.ensembl import write_ensembl_exports
from phibase_pipeline.migrate import make_combined_export
from phibase_pipeline.validate import validate_export


parser = argparse.ArgumentParser(
    prog='phibase-pipeline',
    description='Pipeline for version 5 of the PHI-base database.',
)
subparsers = parser.add_subparsers(dest='target', required=True)

parser_zenodo = subparsers.add_parser('zenodo')
parser_zenodo.add_argument(
    'phibase',
    metavar='PHIBASE_CSV',
    type=str,
    help='the path to the PHI-base 4 CSV export file',
)
parser_zenodo.add_argument(
    'phicanto',
    metavar='CANTO_JSON',
    type=str,
    help='the path to the PHI-Canto JSON export file',
)
parser_zenodo.add_argument(
    'output',
    metavar='OUTFILE',
    type=str,
    help='the path to write the combined JSON export file',
)

parser_ensembl = subparsers.add_parser('ensembl')
parser_ensembl.add_argument(
    'phibase',
    metavar='PHIBASE_CSV',
    type=str,
    help='the path to the PHI-base 4 CSV export file',
)
parser_ensembl.add_argument(
    'phicanto',
    metavar='CANTO_JSON',
    type=str,
    help='the path to the PHI-Canto JSON export file',
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

args = parser.parse_args()

if args.target == 'zenodo':
    canto_json = make_combined_export(args.phibase, args.phicanto)
    validate_export(canto_json)
    with open(args.output, 'w+', encoding='utf8') as json_file:
        json.dump(canto_json, json_file, indent=4, sort_keys=True, ensure_ascii=False)
elif args.target == 'ensembl':
    write_ensembl_exports(
        phibase_path=args.phibase,
        canto_export_path=args.phicanto,
        uniprot_data_path=args.uniprot_data,
        dir_path=args.out_dir,
    )
else:
    # argparse should prevent this from being reached
    raise ValueError(f'unsupported target type: {args.target}')
