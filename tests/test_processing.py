#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define the unittests for data cube timeline."""

import os

from cube_builder.config import Config
from cube_builder.utils.processing import build_cube_path


def assert_data_cube_path(datacube, period, tile_id, version, expected_base_path, band=None, prefix=Config.DATA_DIR):
    """Assert directive to validate data cube paths."""
    absolute_datacube_path = build_cube_path(datacube, period, tile_id, version, band=band, prefix=prefix)
    version_str = 'v{0:03d}'.format(version)

    expected_path = os.path.join(prefix, expected_base_path, datacube, version_str, tile_id, period)
    expected_file_name = f'{datacube}_{version_str}_{tile_id}_{period}_{band}.tif'

    assert str(absolute_datacube_path.parent) == expected_path
    assert absolute_datacube_path.name == expected_file_name


def test_datacube_paths():
    """Test directive for data cube base paths."""
    date = '2017-01-01'
    period = f'{date}_2017-01-31'
    tile_id = '000000'
    version = 2
    band = 'b001'
    datacube = 'MyCube_10'

    for prefix in [Config.DATA_DIR, Config.WORK_DIR]:
        # Identity
        assert_data_cube_path(datacube, date, tile_id, version, 'identity', band=band, prefix=prefix)
        # Composed
        assert_data_cube_path(f'{datacube}_1M_MED', period, tile_id, version, 'composed', band=band, prefix=prefix)
