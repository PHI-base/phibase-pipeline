# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

"""Functions to interface with the ROBOT tool https://robot.obolibrary.org/ using
the subprocess module."""

import json
import shutil
import subprocess
import warnings

from pathlib import Path
from tempfile import TemporaryDirectory


def autodetect_robot_path():
    robot_path = shutil.which('robot')  # expects to get the path to robot.bat
    if robot_path is None:
        return None
    dir_path = Path(robot_path).parent
    jar_path = dir_path / 'robot.jar'
    return jar_path


def get_ontology_terms_and_labels(term_label_mapping_config, robot_jar_path=None):

    def has_no_labels(labels):
        return not labels or all(x == '' for x in labels)

    robot_jar_path = robot_jar_path or autodetect_robot_path()
    if robot_jar_path is None:
        warnings.warn(
            'path to robot.jar is not specified and cannot be autodetected in PATH. Not including ontology term labels.'
        )
        return {}
    if not term_label_mapping_config:
        return {}
    process_args = {
        'java_command': 'java',
        'java_opts': '-jar',
        'jar_path': str(robot_jar_path),
        'command_name': 'export',
        'input_option': '--input',
        'input_value': None,
        'header_option': '--header',
        'header_value': 'ID|LABEL',
        'format_option': '--format',
        'format_value': 'json',
        'export_option': '--export',
        'export_value': None,
    }
    term_label_mapping = {}
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)  # TemporaryDirectory does not yet return a Path object
        for prefix, path in term_label_mapping_config.items():
            output_path = str(output_dir / f'{prefix}_export.json')
            process_args['input_value'] = str(path)
            process_args['export_value'] = output_path
            args = list(process_args.values())
            completed_process = subprocess.run(args, encoding='utf-8')
            completed_process.check_returncode()
            with open(output_path, encoding='utf-8') as output_file:
                results = json.load(output_file)
            for result in results:
                term_id = result['ID'].replace('obo:', '').replace('_', ':')
                labels = result['LABEL']
                # check current prefix to only export terms from the active ontology
                current_prefix = term_id.split(':')[0]
                if current_prefix != prefix or has_no_labels(labels):
                    continue
                term_label_mapping[term_id] = labels[0]  # we only need one label
    return term_label_mapping
