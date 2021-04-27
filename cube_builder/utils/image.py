#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define a utility to validate merge images."""

import logging
from collections import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import numpy
import rasterio
from rasterio._warp import Affine
from sqlalchemy.engine.result import ResultProxy, RowProxy

from ..config import Config
from .processing import SmartDataSet, generate_cogs, save_as_cog

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


def match_histogram_with_merges(source: str, source_mask: str, reference: str, reference_mask: str, block_size: int = None):
    """Normalize the source image histogram with reference image.

    This functions implements the `skimage.exposure.match_histograms`, which consists in the manipulate the pixels of an
    input image and match the histogram with the reference image.

    See more in `Histogram Matching <https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html>`_.

    Note:
        It overwrites the source file.

    Args:
        source (str): Path to the rasterio data set file
        source_mask (str): Path to the rasterio data set file
        reference (str): Path to the rasterio data set file
        reference_mask (str): Path to the rasterio data set file
    """
    from skimage.exposure import match_histograms as _match_histograms

    with rasterio.open(source) as source_data_set, rasterio.open(source_mask) as source_mask_data_set:
        source_arr = source_data_set.read(1, masked=True)
        source_mask_arr = source_mask_data_set.read(1)
        source_options = source_data_set.profile.copy()

    with rasterio.open(reference) as reference_data_set, rasterio.open(reference_mask) as reference_mask_data_set:
        reference_arr = reference_data_set.read(1, masked=True)
        reference_mask_arr = reference_mask_data_set.read(1)

    intersect_mask = numpy.logical_and(
        source_mask_arr < 255,  # CHECK: Use only valid data? numpy.isin(source_mask_arr, [0, 1, 3]),
        reference_mask_arr < 255,   # CHECK: Use only valid data? numpy.isin(reference_mask_arr, [0, 1, 3]),
    )

    valid_positions = numpy.where(intersect_mask)

    if valid_positions and len(valid_positions[0]) == 0:
        return

    intersected_source_arr = source_arr[valid_positions]
    intersected_reference_arr = reference_arr[valid_positions]

    histogram = _match_histograms(intersected_source_arr, intersected_reference_arr)

    histogram = numpy.round(histogram).astype(source_options['dtype'])

    source_arr[valid_positions] = histogram

    save_as_cog(str(source), source_arr, block_size=block_size, mode='w', **source_options)


def radsat_extract_bits(bit_value: Union[int, numpy.ndarray], bit_start: int, bit_end: Optional[int] = None):
    """Extract bitwise values from image.

    This method uses the bitwise operation to identify pixel saturation.
    According to the document `LaSRC Product Guide <https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1368_L8_C1-LandSurfaceReflectanceCode-LASRC_ProductGuide-v3.pdf>`_,
    the Landsat Radiometric Saturation Quality Assessment Band (radsat_qa) is a bit
    packed representation of which sensor bands were saturated during data sensing capture.
    The value 1 represents saturated value while 0 is valid data.
    For Landsat-8, the following table represents pixels saturation::

        Bit    Bit Value    Description
          0        1        Data Fill Flag
          1        2        Band 1 Data Saturation Flag
          2        4        Band 2 Data Saturation Flag
          3        8        Band 3 Data Saturation Flag
          4       16        Band 4 Data Saturation Flag
          5       32        Band 5 Data Saturation Flag
          6       64        Band 6 Data Saturation Flag
          7      128        Band 7 Data Saturation Flag
          8      256        Band 8 Data Saturation Flag
          9      512        Band 9 Data Saturation Flag
         10     1024        Band 10 Data Saturation Flag
         11     2048        Band 11 Data Saturation Flag

    Example:
        >>> from cube_builder.utils.image import radsat_extract_bits
        >>> # Represents band 10 (1024) and band 1 (2) is saturated.
        >>> # Check if any band is saturated
        >>> radsat_extract_bits(1026, 1, 7)
        1
        >>> # You can also pass the numpy array
        >>> # radsat_extract_bits(numpy.random.randint(0, 1028, size=(100, 100)), 1, 7)
    """
    if bit_end is None:
        bit_end = bit_start

    mask_size = (1 + bit_end) - bit_start
    mask = (1 << mask_size) - 1

    res = (bit_value >> bit_start) & mask

    return res


def extract_qa_bits(band_data, bit_location, bit_length) -> numpy.ma.masked_array:
    """Extract Quality Assessment Bits from Landsat-8 Collection 2 Level-2 products.

    This method uses the bitwise operation to extract bits according to the document
    `https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1619_Landsat8-C2-L2-ScienceProductGuide-v2.pdf`, page 13.

    Notes:
        Currently this method masks the value that has any bit with Medium or High Confidence.
        It will be support any confidence value in the next releases.

    Args:
        band_data (numpy.ma.masked_array) - The QA Raster Data
        bit_location (int) - The band bit value
        bit_length (int) - How much bits should match.

    Returns:
        numpy.ma.masked_array An array with masked values.
    """
    # Bit shift offset for value comparison
    value = 1

    for bit_offset in [8, 10, 12, 14]:  # 9~8 => Cloud Confidence
        shifted_data = (band_data >> bit_offset) & 2 > 0

        band_data.mask += shifted_data

    position_value = bit_length << bit_location
    shifted_value = value << bit_location

    mask = (band_data & position_value) == shifted_value
    # Only mask the unmatched values
    band_data.mask = numpy.invert(mask)

    return band_data


def get_qa_mask(data: numpy.ma.masked_array, clear_data: List[float] = None, nodata: float = None) -> numpy.ma.masked_array:
    """Get the Raster Mask from any Landsat Quality Assessment product."""
    is_numpy_or_masked_array = type(data) in (numpy.ndarray, numpy.ma.masked_array)
    if type(data) in (float, int,):
        data = numpy.ma.masked_array([data])
    elif (isinstance(data, Iterable) and not is_numpy_or_masked_array) or (isinstance(data, numpy.ndarray) and not hasattr(data, 'mask')):
        data = numpy.ma.masked_array(data, mask=data == nodata, fill_value=nodata)
    elif not is_numpy_or_masked_array:
        raise TypeError(f'Expected a number or numpy masked array for {data}')

    result = data
    internal = numpy.invert(data.astype(numpy.bool_))

    for value in clear_data:
        matched = extract_qa_bits(data, value, 1)

        internal += numpy.invert(matched.mask)

    if nodata is not None:
        data[data == nodata] = numpy.ma.masked

    data.mask = numpy.invert(internal)

    return result
