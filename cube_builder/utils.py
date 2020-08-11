#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define celery tasks utilities for datacube generation."""

# Python Native
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

# 3rdparty
import numpy
import rasterio
from bdc_db.models import Asset, Band, CollectionItem, db
from flask import abort
from numpngw import write_png
from rasterio import Affine, MemoryFile
from rasterio.warp import Resampling, reproject

# BDC Scripts
from .config import Config


# Constant to define required bands to generate both NDVI and EVI
from .constants import CLEAR_OBSERVATION_ATTRIBUTES, PROVENANCE_NAME, TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, \
    PROVENANCE_ATTRIBUTES

VEGETATION_INDEX_BANDS = {'red', 'nir', 'blue'}


def get_rasterio_config() -> dict:
    """Retrieve cube-builder global config for the rasterio module."""
    options = dict()

    if Config.RASTERIO_ENV and isinstance(Config.RASTERIO_ENV, dict):
        options.update(Config.RASTERIO_ENV)

    return options


def get_or_create_model(model_class, defaults=None, **restrictions):
    """Define a utility method for looking up an object with the given restrictions, creating one if necessary.

    Args:
        model_class (BaseModel) - Base Model of Brazil Data Cube DB
        defaults (dict) - Values to fill out model instance
        restrictions (dict) - Query Restrictions
    Returns:
        BaseModel Retrieves model instance
    """
    instance = db.session.query(model_class).filter_by(**restrictions).first()

    if instance:
        return instance, False

    params = dict((k, v) for k, v in restrictions.items())

    params.update(defaults or {})
    instance = model_class(**params)

    db.session.add(instance)

    return instance, True


def get_or_create_activity(cube: str, warped: str, activity_type: str, scene_type: str, band: str,
                           period: str, tile_id: str, activity_date: str, **parameters):
    """Define a utility method for create activity."""
    return dict(
        band=band,
        collection_id=cube,
        warped_collection_id=warped,
        activity_type=activity_type,
        tags=parameters.get('tags', []),
        status='CREATED',
        date=activity_date,
        period=period,
        scene_type=scene_type,
        args=parameters,
        tile_id=tile_id
    )


class DataCubeFragments(list):
    """Parse a data cube name and retrieve their parts.

    A data cube is composed by the following structure:
    ``Collections_Resolution_TemporalPeriod_CompositeFunction``.

    An IDENTITY data cube does not have TemporalPeriod and CompositeFunction.

    Examples:
        >>> # Parse Sentinel 2 Monthly MEDIAN
        >>> cube_parts = DataCubeFragments('S2_10_1M_MED') # ['S2', '10', '1M', 'MED']
        >>> cube_parts.composite_function
        ... 'MED'
        >>> # Parse Sentinel 2 IDENTITY
        >>> cube_parts = DataCubeFragments('S2_10') # ['S2', '10']
        >>> cube_parts.composite_function
        ... 'IDENTITY'
        >>> DataCubeFragments('S2-10') # ValueError Invalid data cube name
    """

    def __init__(self, datacube: str):
        """Construct a Data Cube Fragments parser.

        Exceptions:
            ValueError when data cube name is invalid.
        """
        cube_fragments = self.parse(datacube)

        self.datacube = '_'.join(cube_fragments)

        super(DataCubeFragments, self).__init__(cube_fragments)

    @staticmethod
    def parse(datacube: str) -> List[str]:
        cube_fragments = datacube.split('_')

        if len(cube_fragments) > 5 or len(cube_fragments) < 3:
            abort(400, 'Invalid data cube name. "{}"'.format(datacube))

        return cube_fragments

    def __str__(self):
        """Retrieve the data cube name."""
        return self.datacube

    @property
    def composite_function(self):
        """Retrieve data cube composite function based.

        TODO: Add reference to document User Guide - Convention Data Cube Names
        """
        if len(self) < 5:
            return 'IDENTITY'

        return self[-2]


def get_cube_parts(datacube: str) -> DataCubeFragments:
    """Builds a `DataCubeFragments` and validate data cube name policy."""
    return DataCubeFragments(datacube)


def get_cube_id(datacube: str, func=None):
    """Prepare data cube name based on temporal function."""
    cube_fragments = get_cube_parts(datacube)

    if not func or func.upper() == 'IDENTITY':
        return f"{'_'.join(cube_fragments[:2])}_{cube_fragments[-1]}"

    # Ensure that data cube with composite function must have a
    # temporal resolution
    if len(cube_fragments) <= 3:
        raise ValueError('Invalid cube id without temporal resolution. "{}"'.format(datacube))

    # When data cube have temporal resolution (S2_10_1M) use it
    # Otherwise, just remove last cube part (composite function)
    cube = datacube if len(cube_fragments) == 4 else '_'.join(cube_fragments[:-2])

    return f'{cube}_{func}_{cube_fragments[-1]}'


def save_as_cog(destination: str, raster, mode='w', **profile):
    """Save the raster file as Cloud Optimized GeoTIFF.

    See Also:
        Cloud Optimized GeoTiff https://gdal.org/drivers/raster/cog.html

    Args:
        destination: Path to store the data set.
        raster: Numpy raster values to persist in disk
        mode: Default rasterio mode. Default is 'w' but you also can set 'r+'.
        **profile: Rasterio profile values to add in dataset.
    """
    with rasterio.open(str(destination), mode, **profile) as dataset:
        if profile.get('nodata'):
            dataset.nodata = profile['nodata']

        dataset.write_band(1, raster)
        dataset.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
        dataset.update_tags(ns='rio_overview', resampling='nearest')


def merge(merge_file: str, assets: List[dict], band_map, **kwargs):
    """Apply datacube merge scenes.

    TODO: Describe how it works.

    Args:
        merge_file: Path to store data cube merge
        assets: List of collections assets during period
        bands: List of bands from current merge date.
        **kwargs: Extra properties
    """
    nodata = kwargs.get('nodata', -9999)
    xmin = kwargs.get('xmin')
    ymax = kwargs.get('ymax')
    dist_x = kwargs.get('dist_x')
    dist_y = kwargs.get('dist_y')
    dataset = kwargs.get('dataset')
    band = assets[0]['band']
    resx, resy = kwargs.get('resx'), kwargs.get('resy')

    num_pixel_x = round(dist_x / resx)
    num_pixel_y = round(dist_y / resy)
    new_res_x = dist_x / num_pixel_x
    new_res_y = dist_y / num_pixel_y

    cols = num_pixel_x
    rows = num_pixel_y

    srs = kwargs['srs']

    transform = Affine(new_res_x, 0, xmin, 0, -new_res_y, ymax)

    is_sentinel_landsat_quality_fmask = ('LC8SR' in dataset or 'S2_MSI' in dataset) and band == band_map['quality']
    source_nodata = 0

    if band == band_map['quality']:
        resampling = Resampling.nearest

        nodata = 0

        # TODO: Remove it when a custom mask feature is done
        # Identifies when the collection is Sentinel or Landsat
        # In this way, we must keep in mind that fmask 4.2 uses 0 as valid value and 255 for nodata. So, we need
        # to track the dummy data in re-project step in order to prevent store "nodata" as "valid" data (0).
        if is_sentinel_landsat_quality_fmask:
            nodata = 255  # temporally set nodata to 255 in order to reproject without losing valid 0 values
            source_nodata = nodata

        raster = numpy.zeros((rows, cols,), dtype=numpy.uint16)
        raster_merge = numpy.full((rows, cols,), dtype=numpy.uint16, fill_value=source_nodata)
        raster_mask = numpy.ones((rows, cols,), dtype=numpy.uint16)
    else:
        resampling = Resampling.bilinear
        raster = numpy.zeros((rows, cols,), dtype=numpy.int16)
        raster_merge = numpy.full((rows, cols,), fill_value=nodata, dtype=numpy.int16)

    template = None

    with rasterio.Env(CPL_CURL_VERBOSE=False, **get_rasterio_config()):
        for asset in assets:
            with rasterio.open(asset['link']) as src:
                meta = src.meta.copy()
                meta.update({
                    'crs': srs,
                    'transform': transform,
                    'width': cols,
                    'height': rows
                })

                if src.profile['nodata'] is not None:
                    source_nodata = src.profile['nodata']
                elif 'LC8SR' in dataset:
                    if band != band_map['quality']:
                        # Temporary workaround for landsat
                        # Sometimes, the laSRC does not generate the data set properly and
                        # the data maybe UInt16 instead Int16
                        source_nodata = nodata if src.profile['dtype'] == 'int16' else 0
                elif 'CBERS' in dataset and band != band_map['quality']:
                    source_nodata = nodata

                kwargs.update({
                    'nodata': source_nodata
                })

                with MemoryFile() as mem_file:
                    with mem_file.open(**meta) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=raster,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=srs,
                            src_nodata=source_nodata,
                            dst_nodata=nodata,
                            resampling=resampling)

                        if band != 'quality' or is_sentinel_landsat_quality_fmask:
                            valid_data_scene = raster[raster != nodata]
                            raster_merge[raster != nodata] = valid_data_scene.reshape(numpy.size(valid_data_scene))
                        else:
                            raster_merge = raster_merge + raster * raster_mask
                            raster_mask[raster != nodata] = 0

                        if template is None:
                            template = dst.profile
                            # Ensure type is >= int16

                            if band != band_map['quality']:
                                template['dtype'] = 'int16'
                                template['nodata'] = nodata

    # Evaluate cloud cover and efficacy if band is quality
    efficacy = 0
    cloudratio = 100
    if band == band_map['quality']:
        raster_merge, efficacy, cloudratio = getMask(raster_merge, dataset)
        template.update({'dtype': 'uint8'})
        nodata = 255

    template['nodata'] = nodata

    # Ensure file tree is created
    Path(merge_file).parent.mkdir(parents=True, exist_ok=True)

    template.update({
        'compress': 'LZW',
        'tiled': True,
        "interleave": "pixel",
    })

    # Persist on file as Cloud Optimized GeoTIFF
    save_as_cog(str(merge_file), raster_merge, **template)

    return dict(
        file=str(merge_file),
        efficacy=efficacy,
        cloudratio=cloudratio,
        dataset=dataset,
        resolution=resx,
        nodata=nodata
    )


def post_processing_quality(quality_file: str, bands: List[str], cube: str,
                            date: str, tile_id, quality_band: str):
    """Stack the merge bands in order to apply a filter on the quality band.

    We have faced some issues regarding `nodata` value in spectral bands, which was resulting
    in wrong provenance date on STACK data cubes, since the Fmask tells the pixel is valid (0) but a nodata
    value is found in other bands.
    To avoid that, we read all the others bands, seeking for `nodata` value. When found, we set this to
    nodata in Fmask output::

        Quality             Nir                   Quality

        0 0 2 4      702  876 7000 9000      =>    0 0 2 4
        0 0 0 0      687  987 1022 1029      =>    0 0 0 0
        0 2 2 4    -9999 7100 7322 9564      =>  255 2 2 4

    Args:
         quality_file: Path to the merge fmask.
         bands: All the bands from the merge date.
         cube: IDENTITY data cube name
         date: Merge date
         tile_id: Brazil data cube tile identifier
         quality_band: Quality band name
    """
    # Get quality profile and chunks
    with rasterio.open(str(quality_file)) as merge_dataset:
        blocks = list(merge_dataset.block_windows())
        profile = merge_dataset.profile
        nodata = profile.get('nodata', 255)
        raster_merge = merge_dataset.read(1)

    bands_without_quality = [b for b in bands if b != quality_band and b.lower() not in ('ndvi', 'evi', 'cnc', TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME)]

    for _, block in blocks:
        nodata_positions = []

        row_offset = block.row_off + block.height
        col_offset = block.col_off + block.width

        for band in bands_without_quality:
            band_file = build_cube_path(get_cube_id(cube), date, tile_id, band=band)

            with rasterio.open(str(band_file)) as ds:
                raster = ds.read(1, window=block)

            nodata_found = numpy.where(raster == -9999)
            raster_nodata_pos = numpy.ravel_multi_index(nodata_found, raster.shape)
            nodata_positions = numpy.union1d(nodata_positions, raster_nodata_pos)

        if len(nodata_positions):
            raster_merge[block.row_off: row_offset, block.col_off: col_offset][
                numpy.unravel_index(nodata_positions.astype(numpy.int64), raster.shape)] = nodata

    save_as_cog(str(quality_file), raster_merge, **profile)


class SmartDataSet:
    """Defines utility class to auto close rasterio data set.

    This class is class helper to avoid memory leak of opened data set in memory.
    """

    def __init__(self, file_path: str, mode='r', **properties):
        """Initialize SmartDataSet definition and open rasterio data set."""
        self.path = Path(file_path)
        self.dataset = rasterio.open(file_path, mode=mode, **properties)

    def __del__(self):
        """Close dataset on delete object."""
        self.close()

    def __enter__(self):
        """Use data set context with operator ``with``."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close data set on exit with clause."""
        self.close()

    def close(self):
        """Close rasterio data set."""
        if not self.dataset.closed:
            logging.warning('Closing dataset {}'.format(str(self.path)))

            self.dataset.close()


def compute_data_set_stats(file_path: str) -> Tuple[float, float]:
    """Compute data set efficacy and cloud ratio.

    It opens the given ``file_path`` and calculate the mask statistics, such efficacy and cloud ratio.

    Args:
        file_path - Path to given data set
        data_set_name - Data set name (LC8SR, S2SR_SEN28, CBERS, etc)

    Returns:
        Tuple consisting in efficacy and cloud ratio, respectively.
    """
    with rasterio.open(file_path, 'r') as data_set:
        raster = data_set.read(1)

        efficacy, cloud_ratio = _qa_statistics(raster)

    return efficacy, cloud_ratio


def blend(activity, band_map, build_clear_observation=False):
    """Apply blend and generate raster from activity.

    Basically, the blend operation consists in stack all the images (merges) in period. The stack is based in
    best pixel image (Best clear ratio). The cloud pixels are masked with `numpy.ma` module, enabling to apply
    temporal composite function MEDIAN, AVG over these rasters.

    The following example represents a data cube Landsat-8 16 days using function Best Pixel (Stack - STK) and
    Median (MED) in period of 16 days from 1/1 to 16/1. The images from `10/1` and `15/1` were found and the values as
    described below:

        10/1
        Quality                Nir

        0 0 2 4         702  876 7000 9000
        0 1 1 4         687  444  421 9113      =>  Clear Ratio = 50%
        0 2 2 4        1241 1548 2111 1987      =>  Cloud Ratio = 50%

        15/1
        Quality           Nir
        0 0 255 255     854 756 9800 9454
        0 1   1   1     945 400  402  422       =>  Clear Ratio ~= 83%
        0 0   0   0     869 975  788  799       =>  Cloud Ratio ~= 0%

    According to Brazil Data Cube User Guide, the best image is 15/1 (clear ratio ~83%) and worst as 10/1 (50%).
    The result data cube will be:

        Landsat-8_30_16D_STK
        Quality        Nir                     Provenance (Day of Year)

        0 0 2 4       854 756 7000 9000      15 15 10 10
        0 1 1 1       945 400  411  422      15 15 15 15
        0 0 0 0       869 975  788  799      15 15 15 15

        Landsat-8_30_16D_MED
        Nir

        778  816 -9999 -9999
        816  422   402   422
        1055 975   788   799

    Notes:
        When build_clear_observation is set, make sure to do not execute in parallel processing
        since it is not `thread-safe`.
        The provenance band is not generated by MEDIAN products.
        For pixels `nodata` in the best image, the cube builder will try to find useful pixel in the next observation.
        It may be cloud/cloud-shadow (when there is no valid pixel 0 and 1). Otherwise, fill as `nodata`.

    See Also:
        Numpy Masked Arrays https://numpy.org/doc/stable/reference/maskedarray.generic.html
        TODO: Add Brazil Data Cube User Guide Reference.

    Args:
        activity: Prepared blend activity metadata
        band_map: Map of data cube bands (common_name : name)
        build_clear_observation: Flag to dispatch generation of Clear Observation band. It is not ``thread-safe``.

    Returns:
        A processed activity with the generated values.
    """
    # Assume that it contains a band and quality band
    numscenes = len(activity['scenes'])

    band = activity['band']

    nodata = activity.get('nodata', -9999)
    if band == 'quality':
        nodata = 255

    # Get basic information (profile) of input files
    keys = list(activity['scenes'].keys())

    filename = activity['scenes'][keys[0]]['ARDfiles'][band]

    with rasterio.open(filename) as src:
        profile = src.profile
        tilelist = list(src.block_windows())

    # Order scenes based in efficacy/resolution
    mask_tuples = []

    for key in activity['scenes']:
        scene = activity['scenes'][key]
        resolution = scene.get('resx') or scene.get('resy') or scene.get('resolution')

        efficacy = int(scene['efficacy'])
        resolution = int(resolution)
        mask_tuples.append((100. * efficacy / resolution, key))

    # Open all input files and save the datasets in two lists, one for masks and other for the current band.
    # The list will be ordered by efficacy/resolution
    masklist = []
    bandlist = []

    for m in sorted(mask_tuples, reverse=True):
        key = m[1]
        efficacy = m[0]
        scene = activity['scenes'][key]

        filename = scene['ARDfiles'][band_map['quality']]

        try:
            masklist.append(rasterio.open(filename))
        except BaseException as e:
            raise IOError('FileError while opening {} - {}'.format(filename, e))

        filename = scene['ARDfiles'][band]

        try:
            bandlist.append(rasterio.open(filename))
        except BaseException as e:
            raise IOError('FileError while opening {} - {}'.format(filename, e))

    # Build the raster to store the output images.
    width = profile['width']
    height = profile['height']

    # STACK will be generated in memory
    stack_raster = numpy.full((height, width), dtype=profile['dtype'], fill_value=nodata)
    # Build the stack total observation
    stack_total_observation = numpy.zeros((height, width), dtype=numpy.uint8)

    datacube = activity.get('datacube')
    period = activity.get('period')
    tile_id = activity.get('tile_id')

    cube_file = build_cube_path(datacube, period, tile_id, band, suffix='.tif')

    # Create directory
    cube_file.parent.mkdir(parents=True, exist_ok=True)

    median_raster = numpy.full((height, width), fill_value=nodata, dtype=profile['dtype'])

    if build_clear_observation:
        logging.warning('Creating and computing Clear Observation (ClearOb) file...')

        clear_ob_file_path = build_cube_path(datacube, period, tile_id, band=CLEAR_OBSERVATION_NAME, suffix='.tif')

        clear_ob_profile = profile.copy()
        clear_ob_profile['dtype'] = CLEAR_OBSERVATION_ATTRIBUTES['data_type']
        clear_ob_profile.pop('nodata', None)
        clear_ob_data_set = SmartDataSet(str(clear_ob_file_path), 'w', **clear_ob_profile)

    provenance_array = numpy.full((height, width), dtype=numpy.int16, fill_value=-1)

    for _, window in tilelist:
        # Build the stack to store all images as a masked array. At this stage the array will contain the masked data
        stackMA = numpy.ma.zeros((numscenes, window.height, window.width), dtype=numpy.int16)

        notdonemask = numpy.ones(shape=(window.height, window.width), dtype=numpy.bool_)

        row_offset = window.row_off + window.height
        col_offset = window.col_off + window.width

        # For all pair (quality,band) scenes
        for order in range(numscenes):
            # Read both chunk of Merge and Quality, respectively.
            ssrc = bandlist[order]
            msrc = masklist[order]
            raster = ssrc.read(1, window=window)
            mask = msrc.read(1, window=window)
            copy_mask = numpy.array(mask, copy=True)

            # Mask valid data (0 and 1) as True
            mask[mask < 2] = 1
            # Mask cloud/snow/shadow/no-data as False
            mask[mask >= 2] = 0
            # Ensure that Raster noda value (-9999 maybe) is set to False
            mask[raster == nodata] = 0

            # Create an inverse mask value in order to pass to numpy masked array
            # True => nodata
            bmask = numpy.invert(mask.astype(numpy.bool_))

            # Use the mask to mark the fill (0) and cloudy (2) pixels
            stackMA[order] = numpy.ma.masked_where(bmask, raster)

            # Copy Masked values in order to stack total observation
            copy_mask[copy_mask <= 4] = 1
            copy_mask[copy_mask >= 5] = 0

            stack_total_observation[window.row_off: row_offset, window.col_off: col_offset] += copy_mask.astype(numpy.uint8)

            # Get current observation file name
            file_name = Path(bandlist[order].name).stem
            file_date = file_name.split('_')[4]
            day_of_year = datetime.strptime(file_date, '%Y-%m-%d').timetuple().tm_yday

            # Find all no data in destination STACK image
            stack_raster_where_nodata = numpy.where(
                stack_raster[window.row_off: row_offset, window.col_off: col_offset] == nodata
            )

            # Turns into a 1-dimension
            stack_raster_nodata_pos = numpy.ravel_multi_index(stack_raster_where_nodata,
                                                              stack_raster[window.row_off: row_offset,
                                                              window.col_off: col_offset].shape)

            # Find all valid/cloud in destination STACK image
            raster_where_data = numpy.where(raster != nodata)
            raster_data_pos = numpy.ravel_multi_index(raster_where_data, raster.shape)

            # Match stack nodata values with observation
            # stack_raster_where_nodata && raster_where_data
            intersect_ravel = numpy.intersect1d(stack_raster_nodata_pos, raster_data_pos)

            if len(intersect_ravel):
                where_intersec = numpy.unravel_index(intersect_ravel, raster.shape)
                stack_raster[window.row_off: row_offset, window.col_off: col_offset][where_intersec] = raster[where_intersec]

                provenance_array[window.row_off: row_offset, window.col_off: col_offset][where_intersec] = day_of_year

            # Identify what is needed to stack, based in Array 2d bool
            todomask = notdonemask * numpy.invert(bmask)

            # Find all positions where valid data matches.
            clear_not_done_pixels = numpy.where(numpy.logical_and(todomask, mask.astype(numpy.bool)))

            # Override the STACK Raster with valid data.
            stack_raster[window.row_off: row_offset, window.col_off: col_offset][clear_not_done_pixels] = raster[
                clear_not_done_pixels]

            # Mark day of year to the valid pixels
            provenance_array[window.row_off: row_offset, window.col_off: col_offset][
                clear_not_done_pixels] = day_of_year

            # Update what was done.
            notdonemask = notdonemask * bmask

        median = numpy.ma.median(stackMA, axis=0).data

        median[notdonemask.astype(numpy.bool_)] = nodata

        median_raster[window.row_off: row_offset, window.col_off: col_offset] = median.astype(profile['dtype'])

        if build_clear_observation:
            count_raster = numpy.ma.count(stackMA, axis=0)

            clear_ob_data_set.dataset.write(count_raster.astype(clear_ob_profile['dtype']), window=window, indexes=1)

    # Close all input dataset
    for order in range(numscenes):
        bandlist[order].close()
        masklist[order].close()

    # Evaluate cloud cover
    cloudcover = 100. * ((height * width - numpy.count_nonzero(stack_raster)) / (height * width))

    profile.update({
        'compress': 'LZW',
        'tiled': True,
        'interleave': 'pixel',
    })

    # Since count no cloud operator is specific for a band, we must ensure to manipulate data set only
    # for band clear observation to avoid concurrent processes write same data set in disk.
    # TODO: Review how to design it to avoid these IF's statement, since we must stack data set and mask dummy values
    if build_clear_observation:
        clear_ob_data_set.close()
        logging.warning('Clear Observation (ClearOb) file generated successfully.')

        total_observation_file = build_cube_path(datacube, period, tile_id, band=TOTAL_OBSERVATION_NAME)
        total_observation_profile = profile.copy()
        total_observation_profile.pop('nodata', None)
        total_observation_profile['dtype'] = 'uint8'

        save_as_cog(str(total_observation_file), stack_total_observation, **total_observation_profile)

        with rasterio.open(str(clear_ob_file_path), 'r+', **clear_ob_profile) as dataset:
            dataset.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
            dataset.update_tags(ns='rio_overview', resampling='nearest')

        activity['clear_observation_file'] = str(clear_ob_data_set.path)
        activity['total_observation'] = str(total_observation_file)

    cube_function = DataCubeFragments(datacube).composite_function

    if cube_function == 'MED':
        # Close and upload the MEDIAN dataset
        with rasterio.open(str(cube_file), 'w', **profile) as ds_median:
            ds_median.nodata = nodata
            ds_median.write_band(1, median_raster)
            ds_median.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
            ds_median.update_tags(ns='rio_overview', resampling='nearest')
    else:
        with rasterio.open(str(cube_file), 'w', **profile) as stack_dataset:
            stack_dataset.nodata = nodata
            stack_dataset.write_band(1, stack_raster)
            stack_dataset.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
            stack_dataset.update_tags(ns='rio_overview', resampling='nearest')

        if build_clear_observation:
            provenance_file = build_cube_path(datacube, period, tile_id, PROVENANCE_NAME)
            provenance_profile = profile.copy()
            provenance_profile.pop('nodata', -1)
            provenance_profile['dtype'] = PROVENANCE_ATTRIBUTES['data_type']

            save_as_cog(str(provenance_file), provenance_array, **provenance_profile)
            activity['provenance'] = str(provenance_file)

    activity['blends'] = {
        cube_function: str(cube_file)
    }

    activity['efficacy'] = efficacy
    activity['cloudratio'] = cloudcover

    return activity


def generate_rgb(rgb_file: Path, qlfiles: List[str]):
    # TODO: Save RGB definition on Database
    with rasterio.open(str(qlfiles[0])) as dataset:
        profile = dataset.profile

    profile['count'] = 3
    with rasterio.open(str(rgb_file), 'w', **profile) as dataset:
        for band_index in range(len(qlfiles)):
            with rasterio.open(str(qlfiles[band_index])) as band_dataset:
                data = band_dataset.read(1)
                dataset.write(data, band_index + 1)

    logging.info(f'Done RGB {str(rgb_file)}')


def has_vegetation_index_support(scene_bands: List[str], band_map: dict):
    """Check if can generate NDVI and EVI bands.

    This method tries to check if the cube execution has all required bands
    to generate both Normalized Difference Vegetation Index (NDVI) and
    Enhanced Vegetation Index (EVI).

    Args:
        scene_bands: Data cube bands names generated in current execution
        band_map: All data cube bands, mapped by `band_common_name`: `band_name`.
    """
    common_name_bands = set()

    for common_name, name in band_map.items():
        if name in scene_bands:
            common_name_bands.add(common_name)

    ndvi_in_band_map = 'ndvi' in band_map or 'NDVI' in band_map
    evi_in_band_map = 'evi' in band_map or 'EVI' in band_map

    return (ndvi_in_band_map and evi_in_band_map) \
        and VEGETATION_INDEX_BANDS <= set(common_name_bands)


def publish_datacube(cube, bands, datacube, tile_id, period, scenes, cloudratio, band_map, **kwargs):
    """Generate quicklook and catalog datacube on database."""
    start_date, end_date = period.split('_')

    raster_size_schemas = cube.raster_size_schemas

    cube_parts = get_cube_parts(cube.id)

    for composite_function in [cube_parts.composite_function]:
        item_datacube = get_cube_id(cube.id, composite_function)

        item_id = '{}_{}_{}'.format(item_datacube, tile_id, period)

        _datacube = get_cube_id(datacube, composite_function)

        cube_bands = Band.query().filter(Band.collection_id == _datacube).all()

        quick_look_file = build_cube_path(_datacube, period, tile_id, suffix=None)

        ql_files = []
        for band in bands:
            ql_files.append(scenes[band][composite_function])

        quick_look_file = generate_quick_look(str(quick_look_file), ql_files)

        if kwargs.get('with_rgb'):
            rgb_file = build_cube_path(cube.id, period, tile_id, 'RGB')
            generate_rgb(rgb_file, ql_files)

        if has_vegetation_index_support(scenes.keys(), band_map):
            # Generate VI
            evi_name = band_map.get('evi', 'EVI')
            evi_path = build_cube_path(cube.id, period, tile_id, band=evi_name)
            ndvi_name = band_map.get('ndvi', 'NDVI')
            ndvi_path = build_cube_path(cube.id, period, tile_id, band=ndvi_name)

            generate_evi_ndvi(scenes[band_map['red']][composite_function],
                              scenes[band_map['nir']][composite_function],
                              scenes[band_map['blue']][composite_function],
                              str(evi_path), str(ndvi_path))
            scenes[evi_name] = {composite_function: str(evi_path)}
            scenes[ndvi_name] = {composite_function: str(ndvi_path)}

        Asset.query().filter(Asset.collection_item_id == item_id).delete()

        CollectionItem.query().filter(CollectionItem.id == item_id).delete()

        with db.session.begin_nested():
            CollectionItem(
                id=item_id,
                collection_id=item_datacube,
                grs_schema_id=cube.grs_schema_id,
                tile_id=tile_id,
                item_date=start_date,
                composite_start=start_date,
                composite_end=end_date,
                quicklook=quick_look_file.replace(Config.DATA_DIR, ''),
                cloud_cover=cloudratio,
                scene_type=composite_function,
                compressed_file=None
            ).save(commit=False)

            for band in scenes:
                band_model = list(filter(lambda b: b.name == band, cube_bands))

                # Band does not exists on model
                if not band_model:
                    logging.warning('Band {} of {} does not exist on database. Skipping'.format(band, cube.id))
                    continue

                asset_relative_path = scenes[band][composite_function].replace(Config.DATA_DIR, '')

                Asset(
                    collection_id=item_datacube,
                    band_id=band_model[0].id,
                    grs_schema_id=cube.grs_schema_id,
                    tile_id=tile_id,
                    collection_item_id=item_id,
                    url='{}'.format(asset_relative_path),
                    source=None,
                    raster_size_x=raster_size_schemas.raster_size_x,
                    raster_size_y=raster_size_schemas.raster_size_y,
                    raster_size_t=1,
                    chunk_size_x=raster_size_schemas.chunk_size_x,
                    chunk_size_y=raster_size_schemas.chunk_size_y,
                    chunk_size_t=1
                ).save(commit=False)

        db.session.commit()

    return quick_look_file


def publish_merge(bands, datacube, tile_id, period, date, scenes, band_map):
    """Generate quicklook and catalog warped datacube on database.

    TODO: Review it with publish_datacube
    """
    item_id = '{}_{}_{}'.format(datacube.id, tile_id, date)

    quick_look_file = build_cube_path(datacube.id, date, tile_id, suffix=None)

    cube_bands = Band.query().filter(Band.collection_id == datacube.id).all()
    raster_size_schemas = datacube.raster_size_schemas

    ql_files = []
    for band in bands:
        ql_files.append(scenes['ARDfiles'][band])

    quick_look_file = generate_quick_look(str(quick_look_file), ql_files)

    Asset.query().filter(Asset.collection_item_id == item_id).delete()

    CollectionItem.query().filter(CollectionItem.id == item_id).delete()

    # Generate VI
    if has_vegetation_index_support(scenes['ARDfiles'].keys(), band_map):
        evi_name = band_map.get('evi', 'EVI')
        evi_path = build_cube_path(datacube.id, date, tile_id, band=evi_name)
        ndvi_name = band_map.get('ndvi', 'NDVI')
        ndvi_path = build_cube_path(datacube.id, date, tile_id, band=ndvi_name)

        generate_evi_ndvi(scenes['ARDfiles'][band_map['red']],
                          scenes['ARDfiles'][band_map['nir']],
                          scenes['ARDfiles'][band_map['blue']],
                          str(evi_path), str(ndvi_path))
        scenes['ARDfiles'][evi_name] = str(evi_path)
        scenes['ARDfiles'][ndvi_name] = str(ndvi_path)

    with db.session.begin_nested():
        CollectionItem(
            id=item_id,
            collection_id=datacube.id,
            grs_schema_id=datacube.grs_schema_id,
            tile_id=tile_id,
            item_date=date,
            composite_start=date,
            composite_end=period.split('_')[-1],
            quicklook=quick_look_file.replace(Config.DATA_DIR, ''),
            cloud_cover=scenes.get('cloudratio', 0),
            scene_type='WARPED',
            compressed_file=None
        ).save(commit=False)

        for band in scenes['ARDfiles']:
            band_model = list(filter(lambda b: b.name == band, cube_bands))

            # Band does not exists on model
            if not band_model:
                logging.warning('Band {} of {} does not exist on database'.format(band, datacube.id))
                continue

            asset_relative_path = scenes['ARDfiles'][band].replace(Config.DATA_DIR, '')

            Asset(
                collection_id=datacube.id,
                band_id=band_model[0].id,
                grs_schema_id=datacube.grs_schema_id,
                tile_id=tile_id,
                collection_item_id=item_id,
                url='{}'.format(asset_relative_path),
                source=None,
                raster_size_x=raster_size_schemas.raster_size_x,
                raster_size_y=raster_size_schemas.raster_size_y,
                raster_size_t=1,
                chunk_size_x=raster_size_schemas.chunk_size_x,
                chunk_size_y=raster_size_schemas.chunk_size_y,
                chunk_size_t=1
            ).save(commit=False)

    db.session.commit()

    return quick_look_file


def generate_quick_look(file_path, qlfiles):
    """Generate quicklook on disk."""
    with rasterio.open(qlfiles[0]) as src:
        profile = src.profile

    numlin = 768
    numcol = int(float(profile['width'])/float(profile['height'])*numlin)
    image = numpy.ones((numlin,numcol,len(qlfiles),), dtype=numpy.uint8)
    pngname = '{}.png'.format(file_path)

    nb = 0
    for file in qlfiles:
        with rasterio.open(file) as src:
            raster = src.read(1, out_shape=(numlin, numcol))

            # Rescale to 0-255 values
            nodata = raster <= 0
            if raster.min() != 0 or raster.max() != 0:
                raster = raster.astype(numpy.float32)/10000.*255.
                raster[raster > 255] = 255
            image[:, :, nb] = raster.astype(numpy.uint8) * numpy.invert(nodata)
            nb += 1

    write_png(pngname, image, transparent=(0, 0, 0))
    return pngname


def getMask(raster, dataset):
    """Retrieve and re-sample quality raster to well-known values used in Brazil Data Cube.

    We adopted the `Fmask <https://github.com/GERSL/Fmask>`_ (Function of Mask).
    TODO: Add paper/authors reference

    In the Fmask output product, the following classes are presented in a normative quality:
        - 0: Clear Land Pixel
        - 1: Clear Water Pixel
        - 2: Cloud Shadow
        - 3: Snow-ice
        - 4: Cloud
        - 255: no observation

    For satellite which does not supports these values, consider to expose individual values.
    For example:
        CBERS does not have `Snow-ice`, `Water` or `Cloud Shadow` pixel values. The following values are described
        in CBERS quality band:
        - `0`: Fill/Nodata. Re-sample to `No observation` (255);
        - `127: Valid Data. Re-sample to `Clear Land Pixel` (0);
        - `255`: Cloudy. Re-sample to `Cloud` (4)

    Args:
        raster - Raster with Quality band values
        satellite - Satellite Type

    Returns:
        Tuple containing formatted quality raster, efficacy and cloud ratio, respectively.
    """
    rastercm = raster
    if dataset == 'MOD13Q1' or dataset == 'MYD13Q1':
        # MOD13Q1 Pixel Reliability !!!!!!!!!!!!!!!!!!!!
        # Note that 1 was added to this image in downloadModis because of warping
        # Rank/Key Summary QA 		Description
        # -1 		Fill/No Data 	Not Processed
        # 0 		Good Data 		Use with confidence
        # 1 		Marginal data 	Useful, but look at other QA information
        # 2 		Snow/Ice 		Target covered with snow/ice
        # 3 		Cloudy 			Target not visible, covered with cloud
        lut = numpy.array([255, 0, 0, 2, 4], dtype=numpy.uint8)
        rastercm = numpy.take(lut, raster+1).astype(numpy.uint8)
    elif 'CBERS4' in dataset:
        # Key Summary        QA Description
        #   0 Fill/No Data - Not Processed
        # 127 Good Data    - Use with confidence
        # 255 Cloudy       - Target not visible, covered with cloud
        lut = numpy.zeros(256, dtype=numpy.uint8)
        lut[0] = 255
        lut[127] = 0
        lut[255] = 4
        rastercm = numpy.take(lut, raster).astype(numpy.uint8)

    efficacy, cloudratio = _qa_statistics(rastercm)

    return rastercm.astype(numpy.uint8), efficacy, cloudratio


def _qa_statistics(raster) -> Tuple[float, float]:
    """Retrieve raster statistics efficacy and cloud ratio, based in Fmask values.

    Notes:
        Values 0 and 1 are considered `clear data`.
    """
    totpix = raster.size
    clearpix = numpy.count_nonzero(raster < 2)
    cloudpix = numpy.count_nonzero(raster > 1)
    imagearea = clearpix+cloudpix
    cloudratio = 100
    if imagearea != 0:
        cloudratio = round(100.*cloudpix/imagearea, 1)
    efficacy = round(100.*clearpix/totpix, 2)

    return efficacy, cloudratio


def generate_evi_ndvi(red_band_path: str, nir_band_path: str, blue_bland_path: str, evi_name_path: str, ndvi_name_path: str):
    """Generate Normalized Difference Vegetation Index (NDVI) and Enhanced Vegetation Index (EVI).

    Args:
        red_band_path: Path to the RED band
        nir_band_path: Path to the NIR band
        blue_bland_path: Path to the BLUE band
        evi_name_path: Path to save EVI file
        ndvi_name_path: Path to save NDVI file
    """
    with rasterio.open(nir_band_path) as ds_nir, rasterio.open(red_band_path) as ds_red,\
            rasterio.open(blue_bland_path) as ds_blue:
        profile = ds_nir.profile
        nodata = int(profile['nodata'])
        blocks = ds_nir.block_windows()
        data_type = profile['dtype']

        raster_ndvi = numpy.full((profile['height'], profile['width']), dtype=data_type, fill_value=nodata)
        raster_evi = numpy.full((profile['height'], profile['width']), dtype=data_type, fill_value=nodata)

        for _, block in blocks:
            row_offset = block.row_off + block.height
            col_offset = block.col_off + block.width

            red = ds_red.read(1, masked=True, window=block)
            nir = ds_nir.read(1, masked=True, window=block)

            ndvi_block = (10000. * ((nir - red) / (nir + red))).astype(numpy.int16)
            ndvi_block[ndvi_block == numpy.ma.masked] = nodata

            raster_ndvi[block.row_off: row_offset, block.col_off: col_offset] = ndvi_block

            blue = ds_blue.read(1, masked=True, window=block)

            evi_block = (10000. * 2.5 * (nir - red) / (nir + 6. * red - 7.5 * blue + 10000.)).astype(numpy.int16)
            evi_block[evi_block == numpy.ma.masked] = nodata

            raster_evi[block.row_off: row_offset, block.col_off: col_offset] = evi_block

        save_as_cog(ndvi_name_path, raster_ndvi, **profile)
        save_as_cog(evi_name_path, raster_evi, **profile)


def build_cube_path(datacube: str, period: str, tile_id: str, band: str = None, suffix: Union[str, None] = '.tif') -> Path:
    """Retrieve the path to the Data cube file in Brazil Data Cube Cluster."""
    folder = 'Warped'
    date = period

    fragments = DataCubeFragments(datacube)

    file_name = f'{datacube}_{tile_id}_{period}'

    if band is not None:
        file_name = f'{file_name}_{band}'

    if suffix is None:
        suffix = ''

    file_name = f'{file_name}{suffix}'

    datacube = datacube.replace(f'_{Config.VERSION_PATH_PREFIX}', '')

    if fragments.composite_function != 'IDENTITY':  # For cube with temporal composition
        folder = 'Mosaic'

    return Path(Config.DATA_DIR) / 'Repository' / folder / datacube / Config.VERSION_PATH_PREFIX / tile_id / date / file_name
