import argparse
import json

from phibase_pipeline.migrate import make_combined_export
from phibase_pipeline.validate import validate_export


parser = argparse.ArgumentParser(
    prog='phibase-pipeline',
    description='Pipeline for version 5 of the PHI-base database.',
)
parser.add_argument(
    'phibase',
    metavar='CSV',
    type=str,
    help='the path to the PHI-base 5 CSV export file',
)
parser.add_argument(
    'phicanto',
    metavar='JSON',
    type=str,
    help='the path to the PHI-Canto JSON export file',
)
parser.add_argument(
    'output',
    metavar='OUTFILE',
    type=str,
    help='the path to write the combined JSON export file',
)
args = parser.parse_args()

canto_json = make_combined_export(args.phibase, args.phicanto)
validate_export(canto_json)

with open(args.output, 'w+', encoding='utf8') as json_file:
    json.dump(canto_json, json_file, indent=4, sort_keys=True, ensure_ascii=False)
