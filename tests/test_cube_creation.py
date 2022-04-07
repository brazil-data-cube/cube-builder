#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Test the data cube creation using native interfaces."""

import datetime
import os
from copy import deepcopy
from unittest.mock import patch

import pytest
import rasterio
from celery import chain, group

from cube_builder.celery.tasks import publish
from cube_builder.maestro import Maestro
from cube_builder.utils.image import check_file_integrity

CUBE_PARAMS = dict(
    datacube='LC8-TESTE_30_16D_LCF',
    tiles=['007011'],
    collections=['LC8_SR-1'],
    start_date=datetime.date(year=2020, month=1, day=1),
    end_date=datetime.date(year=2020, month=1, day=16),
    stac_url=os.getenv('STAC_URL', 'https://brazildatacube.dpi.inpe.br/stac/'),
    token=os.getenv('STAC_ACCESS_TOKEN')
)


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
    def test_mock_dispatch_celery(self, mock_prepare_blend, mock_merge, mock_group, mock_chain, maestro):
        maestro.dispatch_celery()
        mock_merge.s.assert_called()
        mock_prepare_blend.s.assert_called_once()
        mock_group.assert_called_with([mock_chain()])

    def test_cube_workflow(self, maestro):
        res = maestro.dispatch_celery()
        band_map = maestro.band_map

        for period in res['blends'].values():
            blend_bands_result = period.apply()
            blend_bands = blend_bands_result.get()

            publish_result = chain(group(blend_bands), publish.s(band_map, **maestro.properties)).apply()

            blend_files, merge_files = publish_result.get()

            cube_stats = set()
            cube_proj4 = set()

            # Validate Rasters
            for entry in blend_files:
                assert check_file_integrity(entry, read_bytes=True)
                if str(entry).endswith('.tif'):
                    with rasterio.open(str(entry)) as ds:
                        # check resolution
                        transform = ds.transform
                        resx, resy, xmin, ymax = transform.a, transform.e, transform.c, transform.f
                        cube_stats.add((resx, resy, xmin, ymax))
                        cube_proj4.add(ds.crs.to_wkt())
            # All files must have same pixel origin and resolution
            assert len(cube_stats) == 1
            # All files must have same proj4
            assert len(cube_proj4) == 1

    def test_cube_workflow_empty_timeline(self, app):
        params = deepcopy(CUBE_PARAMS)
        params['tiles'] = ['035060']
        maestro = Maestro(**params)
        maestro.orchestrate()
        res = maestro.dispatch_celery()
        band_map = maestro.band_map

        for period in res['blends'].values():
            blend_bands_result = period.apply()
            blend_bands = blend_bands_result.get()

            publish_result = chain(group(blend_bands), publish.s(band_map, **maestro.properties)).apply()

            blend_files, merge_files = publish_result.get()

            # Empty timeline must not have published Identity Files
            assert len(merge_files) == 0

            # Validate Rasters
            for entry in blend_files:
                assert check_file_integrity(entry, read_bytes=True)
                # check all == nodata
