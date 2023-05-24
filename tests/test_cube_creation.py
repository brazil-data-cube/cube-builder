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

"""Test the data cube creation using native interfaces."""

import datetime
import os
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest
import rasterio
from celery import chain, group

from cube_builder.celery.tasks import publish
from cube_builder.constants import to_bool
from cube_builder.maestro import Maestro
from cube_builder.utils.image import check_file_integrity

CUBE_PARAMS = dict(
    datacube='LC8-16D-1',
    tiles=['007011'],
    collections=['LC8_SR-1'],
    start_date=datetime.date(year=2020, month=1, day=1),
    end_date=datetime.date(year=2020, month=1, day=16),
    stac_url=os.getenv('STAC_URL', 'https://brazildatacube.dpi.inpe.br/stac/'),
    token=os.getenv('STAC_ACCESS_TOKEN')
)


def _supported_collection(stac_url: str, collections: list, token: str = None, **kwargs) -> bool:
    from cube_builder._adapter import build_stac

    headers = {}
    if token:
        headers['x-api-key'] = token
    stac = build_stac(stac_url, headers=headers)

    for collection in collections:
        if not _get_stac_collection(stac, collection):
            return False

    return True


def _get_stac_collection(stac, collection: str) -> bool:
    try:
        _ = stac.collection(collection)
        return True
    except:
        return False


@pytest.fixture()
def maestro(app) -> Maestro:
    """Create a simple Data Cube Maestro structure."""
    _maestro = Maestro(**CUBE_PARAMS)
    _maestro.orchestrate()
    yield _maestro


class TestCubeCreation:
    """Represent a class to test Data Cube Creation using direct interface cube_builder.maestro.Maestro."""

    def test_maestro(self, app, maestro):
        assert maestro.mosaics
        for tile, context in maestro.mosaics.items():
            assert tile in CUBE_PARAMS['tiles']

            for period, period_context in context['periods'].items():
                start = period_context['start']
                end = period_context['end']
                assert f'{start}_{end}' == period
                assert CUBE_PARAMS['start_date'].strftime('%Y-%m-%d') == start
                assert CUBE_PARAMS['end_date'].strftime('%Y-%m-%d') == end

    @patch(f'cube_builder.maestro.chain')
    @patch(f'cube_builder.maestro.group')
    @patch(f'cube_builder.maestro.warp_merge')
    @patch(f'cube_builder.maestro.prepare_blend')
    def test_mock_run(self, mock_prepare_blend, mock_merge, mock_group, mock_chain, maestro):
        maestro.run()
        mock_merge.s.assert_called()
        mock_prepare_blend.s.assert_called_once()
        mock_group.assert_called_with([mock_chain()])

    @pytest.mark.skipif(not _supported_collection(**CUBE_PARAMS) or to_bool(os.getenv("SKIP_TEST_DATA", "NO")),
                        reason=f"collection is not supported (or skipped) in {CUBE_PARAMS['stac_url']}")
    def test_cube_workflow(self, maestro):
        for blend_files, merge_files in _make_cube(maestro):
            _assert_datacube_period_valid(blend_files, merge_files)

    def test_cube_workflow_empty_timeline(self, app):
        params = deepcopy(CUBE_PARAMS)
        params['tiles'] = ['024016']
        maestro = Maestro(**params)
        maestro.orchestrate()

        for blend_files, merge_files in _make_cube(maestro):
            # Empty timeline must not have published Identity Files
            assert len(merge_files) == 0

            # Validate Rasters
            for entry in blend_files:
                assert check_file_integrity(entry, read_bytes=True)
                # check all == nodata

    @pytest.mark.skipif(to_bool(os.getenv("SKIP_TEST_DATA", "NO")),
                        reason=f"data cube generation of sentinel-2 skipped due env SKIP_TEST_DATA")
    def test_create_cube_sentinel(self, app):
        params = deepcopy(CUBE_PARAMS)
        params['collections'] = ['S2_L2A-1']
        params['datacube'] = 'S2-16D-1'
        params['tiles'] = ['031018', '026029']
        params['skip_vi_identity'] = True  # Disable Vegetation Index for Identity Cubes
        control = Maestro(**params)
        control.orchestrate()

        cube_stats = set()
        cube_proj4 = set()

        for blend_files, merge_files in _make_cube(control):
            period_cube_stats, period_cube_proj4, _ = _assert_datacube_period_valid(blend_files, merge_files)
            cube_proj4 = cube_proj4.union(period_cube_proj4)
            cube_stats = cube_stats.union(period_cube_stats)

        # All tiles MUST HAVE same projection and total pixel/dimension
        assert len(cube_proj4) == 1
        assert len(cube_stats) == 1


def _make_cube(control: Maestro):
    res = control.run()
    band_map = control.band_map

    for period in res['blends'].values():
        blend_bands_result = period.apply()
        blend_bands = blend_bands_result.get()

        publish_result = chain(group(blend_bands), publish.s(band_map, **control.properties)).apply()

        blend_files, merge_files = publish_result.get()

        yield blend_files, merge_files


def _assert_datacube_period_valid(blend_files: Any, merge_files: Any):
    cube_stats = set()
    cube_proj4 = set()
    cube_bounds = set()

    # Validate Rasters
    for entry in blend_files:
        assert check_file_integrity(entry, read_bytes=True)
        if str(entry).endswith('.tif'):
            with rasterio.open(str(entry)) as ds:
                transform = ds.transform
                resx, resy, xmin, ymax = transform.a, transform.e, transform.c, transform.f

                distance_x = ds.bounds[2] - xmin
                distance_y = ymax - ds.bounds[1]
                cube_stats.add((resx, resy, distance_x, distance_y, ds.width, ds.height))
                cube_proj4.add(ds.crs.to_wkt())
                cube_bounds.add(ds.bounds)
    # All files must have same pixel size and total pixels
    assert len(cube_stats) == 1
    # All files must have same proj4
    assert len(cube_proj4) == 1
    # All files must have same geom/bounds
    assert len(cube_bounds) == 1

    return cube_stats, cube_proj4, cube_bounds
