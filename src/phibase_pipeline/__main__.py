# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import sys

from phibase_pipeline import cli

if __name__ == '__main__':
    # First item of args is the script name: skip it
    cli.run(sys.argv[1:])
