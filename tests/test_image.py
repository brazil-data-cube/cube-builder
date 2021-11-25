#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define the unittests for cube_builder.utils.image module."""

from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy
import rasterio
from rasterio.crs import CRS

from cube_builder.utils import image

RASTER_OPTIONS = dict(
    width=100,
    height=100,
    crs=CRS.from_epsg(4326),
    driver='GTiff',
    count=1,
    dtype='int16'
)

RASTER_DATA = numpy.random.randint(100, size=(RASTER_OPTIONS['width'], RASTER_OPTIONS['height'])).astype(numpy.int16)


def test_check_file_integrity():
    """Test the file integrity checking."""
    temp_file = NamedTemporaryFile()
    with rasterio.open(temp_file.name, 'w', **RASTER_OPTIONS) as ds:
        ds.write(RASTER_DATA, 1)

    assert image.check_file_integrity(temp_file.name)
    assert not image.check_file_integrity('/tmp/not-exists.tif')


def test_create_empty_raster():
    """Test the creation of empty data cube item."""
    with TemporaryDirectory() as tmp:
        tmp_file = tmp + '/test.tif'
        xmin, ymax = 6138927.355567569, 10645561.92311954
        resolution = [10, 10]
        dist = [168060.048009797, 109861.84106387943]
        nodata = -9999
        proj4 = '+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +units=m +no_defs '
        image.create_empty_raster(tmp_file, proj4, dtype='int16', xmin=xmin, ymax=ymax,
                                  resolution=resolution, dist=dist, nodata=nodata)

        with rasterio.open(tmp_file) as ds:
            data = ds.read(1)

            assert data.min() == data.max() == nodata
