#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#

"""Define the unittests for data cube timeline."""

import os
from pathlib import Path

from cube_builder.config import Config
from cube_builder.utils.processing import build_cube_path


def assert_data_cube_path(datacube, period, tile_id, version, expected_base_path, band=None, prefix=Config.DATA_DIR,
                          legacy=False):
    """Assert directive to validate data cube paths."""
    format_path_cube = None
    format_item_cube = None
    folder = expected_base_path
    expected_base_path = os.path.join(prefix, expected_base_path)
    expected_path = os.path.join(expected_base_path, datacube.lower(), f'v{version}',
                                 tile_id[:3], tile_id[-3:],
                                 period[:4],
                                 '{0:02d}'.format(int(period[5:7])),
                                 '{0:02d}'.format(int(period[8:10])))
    expected_file_name = f'{datacube.upper()}_V{version}_{tile_id}_{period[:10].replace("-", "")}_{band}.tif'

    if legacy:
        version_str = 'v{0:03d}'.format(version)
        format_path_cube = '{prefix}/{folder}/{datacube}/{version_legacy}/{tile_id}/{period}/{filename}'
        format_item_cube = '{datacube}_v{version_legacy}_{tile_id}_{date}'
        expected_path = os.path.join(expected_base_path, datacube, version_str, tile_id, period)
        expected_file_name = f'{datacube}_{version_str}_{tile_id}_{period}_{band}.tif'

    absolute_datacube_path = build_cube_path(datacube, period, tile_id, version, band=band, prefix=prefix,
                                             format_path_cube=format_path_cube,
                                             format_item_cube=format_item_cube,
                                             composed=folder == 'composed')

    assert str(absolute_datacube_path.parent) == expected_path
    assert absolute_datacube_path.name == expected_file_name


def test_datacube_paths():
    """Test directive for data cube base paths."""
    date = '2017-01-01'
    period = f'{date}_2017-01-31'
    tile_id = '000000'
    version = 2
    band = 'B1'
    datacube = 'MyCube_10'

    for prefix in [Config.DATA_DIR, Config.WORK_DIR]:
        # Identity New / Legacy
        assert_data_cube_path(datacube, date, tile_id, version, 'identity', band=band, prefix=prefix)
        assert_data_cube_path(datacube, date, tile_id, version, 'identity', band=band, prefix=prefix, legacy=True)
        # Composed
        assert_data_cube_path(f'{datacube}-1M', period, tile_id, version, 'composed', band=band, prefix=prefix)
