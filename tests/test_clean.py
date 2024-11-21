# SPDX-FileCopyrightText: 2023-present James Seager <james.seager@rothamsted.ac.uk>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from phibase_pipeline.clean import fix_curation_dates


@pytest.mark.parametrize(
    'dates,expected',
    [
        pytest.param(
            pd.Series(['2005-05-04', '2013-10-01']),
            pd.Series(['2005-05-04', '2013-10-01'], dtype='datetime64[ns]'),
            id='valid_dates',
        ),
        pytest.param(
            pd.Series(['2005-05-04', np.nan, '2013-10-01']),
            pd.Series(['2005-05-04', '2005-05-04', '2013-10-01'], dtype='datetime64[ns]'),
            id='valid_dates_with_gaps',
        ),
        pytest.param(
            pd.Series([39206, '42382', 41365], dtype='object'),
            pd.Series(['2007-05-04', '2016-01-13', '2013-04-01'], dtype='datetime64[ns]'),
            id='serial_dates',
        ),
        pytest.param(
            pd.Series(
                [
                    '04/05/2005',
                    '39206',
                    39206,
                    '2013-10-01',
                    'Nov-16',
                    '20-Feb-17',
                    '17-Jun',
                    'May 2019',
                ],
            ),
            pd.Series(
                [
                    '2005-05-04',
                    '2007-05-04',
                    '2007-05-04',
                    '2013-10-01',
                    '2016-11-01',
                    '2017-02-20',
                    '2017-06-01',
                    '2019-05-01',
                ],
                dtype='datetime64[ns]',
            ),
            id='mixed_dates',
        ),
    ],
)
def test_fix_curation_dates(dates, expected):
    actual = fix_curation_dates(dates)
    assert_series_equal(expected, actual)
