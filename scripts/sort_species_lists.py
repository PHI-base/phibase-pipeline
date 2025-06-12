#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""
Sort the species CSV files in the phibase_pipeline package by the
scientific_name column, in ascending order.
"""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'src' / 'phibase_pipeline' / 'data'

species_files = (
    DATA_DIR / 'phicanto_host_species.csv',
    DATA_DIR / 'phicanto_pathogen_species.csv',
)

for filepath in species_files:
    with open(filepath, mode='r', newline='', encoding='utf-8') as in_file:
        reader = csv.DictReader(in_file)
        sorted_data = sorted(reader, key=lambda row: row['scientific_name'])

    with open(filepath, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sorted_data)
