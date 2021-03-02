#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define a utility to validate merge images."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import rasterio
from rasterio._warp import Affine
from sqlalchemy.engine.result import ResultProxy, RowProxy
from .processing import SmartDataSet, generate_cogs

from ..config import Config

LANDSAT_BANDS = dict(
    int16=['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7', 'evi', 'ndvi'],
    uint16=['pixel_qa']
)


def validate(row: RowProxy):
    """Validate each merge result."""
    url = row.link.replace('chronos.dpi.inpe.br:8089/datastore', 'www.dpi.inpe.br/newcatalog/tmp')

    errors = list()

    if 'File' in row.traceback:
        try:
            with rasterio.open('/vsicurl/{}'.format(url), 'r') as data_set:
                logging.debug('File {} ok'.format(url))

                if row.collection_id.startswith('LC8'):
                    data_type = data_set.meta.get('dtype')

                    band_dtype = LANDSAT_BANDS.get(data_type)

                    if band_dtype is None:
                        errors.append(
                            dict(
                                message='Band {} mismatch with default Landsat 8'.format(row.band),
                                band=row.band,
                                file=url
                            )
                        )

                    file_name = Path(url).stem

                    band_name = file_name.split('_')[8]

                    if band_name not in band_dtype:
                        errors.append(
                            dict(
                                message='Band {} should be {}'.format(row.band, data_type),
                                band=row.band,
                                file=url
                            )
                        )

        except rasterio.RasterioIOError:
            errors.append(dict(message='File not found or invalid.', band=row.band, file=url))

    return row, errors


def validate_merges(images: ResultProxy, num_threads: int = Config.MAX_THREADS_IMAGE_VALIDATOR) -> dict:
    """Validate each merge retrieved from ``Activity.list_merge_files``.

    Args:
        images: Activity merge images
        num_threads: Concurrent processes to validate
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = executor.map(validate, images)

        output = dict()

        for row, errors in futures:
            if row is None:
                continue

            output.setdefault(row.date, dict())
            output[row.date].setdefault('bands', dict())
            output[row.date].setdefault('errors', list())
            output[row.date].setdefault('collections', set())

            output[row.date]['collections'].add(row.data_set)

            output[row.date]['file'] = row.file
            output[row.date]['errors'].extend(errors)

            output[row.date]['bands'].setdefault(row.band, list())
            output[row.date]['bands'][row.band].append(row.link)

        for element in output.values():
            element['collections'] = list(element['collections'])

        return output


def create_empty_raster(location: str, proj4: str, dtype: str, xmin: float, ymax: float,
                        resolution: List[float], dist: List[float], nodata: float, cog=True):
    """Create an data set filled out with nodata.

    This method aims to solve the problem to generate an empty scene to make sure in order to
    follow the data cube timeline.

    Args:
        location (str): Path where file will be generated.
        proj4 (str): Proj4 with Coordinate Reference System.
        dtype (str): Data type
        xmin (float): Image minx (Related to geotransform)
        ymax (float): Image ymax
        resolution (List[float]): Pixel resolution (X, Y)
        dist (List[float]): The distance of X, Y  (Scene offset)
        nodata (float): Scene nodata.
        cog (bool): Flag to generate datacube. Default is `True`.
    """
    resx, resy = resolution
    distx, disty = dist

    cols = round(distx / resx)
    rows = round(disty / resy)

    new_res_x = distx / cols
    new_res_y = disty / rows

    transform = Affine(new_res_x, 0, xmin, 0, -new_res_y, ymax)

    options = dict(
        width=cols,
        height=rows,
        nodata=nodata,
        crs=proj4,
        transform=transform,
        count=1
    )

    ds = SmartDataSet(str(location), mode='w', dtype=dtype, driver='GTiff', **options)

    ds.close()

    if cog:
        generate_cogs(str(location), str(location))

    return str(location)
