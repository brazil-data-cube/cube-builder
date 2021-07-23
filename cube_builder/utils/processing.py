#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define celery tasks utilities for datacube generation."""

# Python Native
import logging
import warnings
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union

# 3rdparty
import numpy
import rasterio
import rasterio.features
import requests
import shapely
import shapely.geometry
from bdc_catalog.models import Item, Tile, db
from bdc_catalog.utils import multihash_checksum_sha256
from flask import abort
from geoalchemy2.shape import from_shape, to_shape
from numpngw import write_png
from rasterio import Affine, MemoryFile
from rasterio.warp import Resampling, reproject
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from ..config import Config
# Constant to define required bands to generate both NDVI and EVI
from ..constants import (CLEAR_OBSERVATION_ATTRIBUTES, CLEAR_OBSERVATION_NAME, COG_MIME_TYPE, DATASOURCE_ATTRIBUTES,
                         DATASOURCE_NAME, PNG_MIME_TYPE, PROVENANCE_ATTRIBUTES, PROVENANCE_NAME, SRID_ALBERS_EQUAL_AREA,
                         TOTAL_OBSERVATION_NAME)
# Builder
from .index_generator import generate_band_indexes

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

    An IDT data cube does not have TemporalPeriod and CompositeFunction.

    Examples:
        >>> # Parse Sentinel 2 Monthly MEDIAN
        >>> cube_parts = DataCubeFragments('S2_10_1M_MED') # ['S2', '10', '1M', 'MED']
        >>> cube_parts.composite_function
        ... 'MED'
        >>> # Parse Sentinel 2 IDENTITY
        >>> cube_parts = DataCubeFragments('S2_10') # ['S2', '10']
        >>> cube_parts.composite_function
        ... 'IDT'
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
        """Parse a data cube name."""
        cube_fragments = datacube.split('_')

        if len(cube_fragments) > 4 or len(cube_fragments) < 2:
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
        if len(self) < 4:
            return 'IDT'

        return self[-1

        ]


def get_cube_parts(datacube: str) -> DataCubeFragments:
    """Build a `DataCubeFragments` and validate data cube name policy."""
    return DataCubeFragments(datacube)


def get_cube_id(datacube: str, func=None):
    """Prepare data cube name based on temporal function."""
    cube_fragments = get_cube_parts(datacube)

    if not func or func.upper() == 'IDT':
        return f"{'_'.join(cube_fragments[:2])}"

    # Ensure that data cube with composite function must have a
    # temporal resolution
    if len(cube_fragments) <= 3:
        raise ValueError('Invalid cube id without temporal resolution. "{}"'.format(datacube))

    # When data cube have temporal resolution (S2_10_1M) use it
    # Otherwise, just remove last cube part (composite function)
    cube = datacube if len(cube_fragments) == 3 else '_'.join(cube_fragments[:-1])

    return f'{cube}_{func}'


def get_item_id(datacube: str, version: int, tile: str, date: str) -> str:
    """Prepare a data cube item structure."""
    version_str = '{0:03d}'.format(version)

    return f'{datacube}_v{version_str}_{tile}_{date}'


def prepare_asset_url(url: str) -> str:
    """Define a simple function to change CBERS url to local INPE provider.

    TODO: Remove it
    """
    from urllib.parse import urljoin, urlparse

    parsed_url = urlparse(url)

    return urljoin(url, parsed_url.path)


def merge(merge_file: str, mask: dict, assets: List[dict], band: str, band_map: dict, quality_band: str, build_provenance=False, compute=False, **kwargs):
    """Apply datacube merge scenes.

    The Merge or Warp consists in a procedure that cropping and mosaicking all imagens that superimpose a target tile
    of common grid, for a specific date.

    See also:
        BDC Warp https://brazil-data-cube.github.io/products/specifications/processing-flow.html#warp-merge-reprojecting-resampling-and-griding

    Args:
        merge_file: Path to store data cube merge
        assets: List of collections assets during period
        band: Merge band name
        band_map: Map of cube band name and common name.
        build_provenance: Build a provenance file for Merge (Used in combined collections)
        **kwargs: Extra properties
    """
    xmin = kwargs.get('xmin')
    ymax = kwargs.get('ymax')
    dist_x = kwargs.get('dist_x')
    dist_y = kwargs.get('dist_y')
    datasets = kwargs.get('datasets')
    resx, resy = kwargs['resx'], kwargs['resy']
    block_size = kwargs.get('block_size')
    shape = kwargs.get('shape', None)

    if shape:
        cols = shape[0]
        rows = shape[1]
    else:
        cols = round(dist_x / resx)
        rows = round(dist_y / resy)

        new_res_x = dist_x / cols
        new_res_y = dist_y / rows

        transform = Affine(new_res_x, 0, xmin, 0, -new_res_y, ymax)

    srs = kwargs['srs']

    if isinstance(datasets, str):
        warnings.warn(
            'Parameter "dataset" got str, expected list of str. It will be deprecated in future.'
        )
        datasets = [datasets]

    source_nodata = nodata = float(band_map[band]['nodata'])
    data_type = band_map[band]['data_type']
    resampling = Resampling.nearest

    if quality_band == band:
        source_nodata = nodata = float(mask['nodata'])
        # Only apply bilinear (change pixel values) for band values
    elif mask.get('saturated_band') != band:
        resampling = Resampling.bilinear

    raster = numpy.zeros((rows, cols,), dtype=data_type)
    raster_merge = numpy.full((rows, cols,), dtype=data_type, fill_value=source_nodata)

    if build_provenance:
        raster_provenance = numpy.full((rows, cols,),
                                       dtype=DATASOURCE_ATTRIBUTES['data_type'],
                                       fill_value=DATASOURCE_ATTRIBUTES['nodata'])

    template = None
    is_combined_collection = len(datasets) > 1

    with rasterio_access_token(kwargs.get('token')) as options:
        with rasterio.Env(CPL_CURL_VERBOSE=False, **get_rasterio_config(), **options):
            for asset in assets:
                link = prepare_asset_url(asset['link'])

                dataset = asset['dataset']

                _check_rio_file_access(link, access_token=kwargs.get('token'))

                with rasterio.open(link) as src:
                    meta = src.meta.copy()
                    meta.update({
                        'width': cols,
                        'height': rows
                    })
                    if not shape:
                        meta.update({
                            'crs': srs,
                            'transform': transform
                        })

                    if src.profile['nodata'] is not None:
                        source_nodata = src.profile['nodata']
                    elif 'LC8SR' in dataset:
                        if band != quality_band:
                            # Temporary workaround for landsat
                            # Sometimes, the laSRC does not generate the data set properly and
                            # the data maybe UInt16 instead Int16
                            source_nodata = nodata if src.profile['dtype'] == 'int16' else 0
                    elif 'CBERS' in dataset and band != quality_band:
                        source_nodata = nodata

                    kwargs.update({
                        'nodata': source_nodata
                    })

                    with MemoryFile() as mem_file:
                        with mem_file.open(**meta) as dst:
                            if shape:
                                raster = src.read(1)
                            else:
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

                            # For combined collections, we must merge only valid data into final data set
                            if is_combined_collection:
                                positions_todo = numpy.where(raster_merge == nodata)

                                if positions_todo:
                                    valid_positions = numpy.where(raster != nodata)

                                    raster_todo = numpy.ravel_multi_index(positions_todo, raster.shape)
                                    raster_valid = numpy.ravel_multi_index(valid_positions, raster.shape)

                                    # Match stack nodata values with observation
                                    # stack_raster_where_nodata && raster_where_data
                                    intersect_ravel = numpy.intersect1d(raster_todo, raster_valid)

                                    if len(intersect_ravel):
                                        where_intersec = numpy.unravel_index(intersect_ravel, raster.shape)
                                        raster_merge[where_intersec] = raster[where_intersec]
                            else:
                                valid_data_scene = raster[raster != nodata]
                                raster_merge[raster != nodata] = valid_data_scene.reshape(numpy.size(valid_data_scene))
                                valid_data_scene = None

                            if template is None:
                                template = dst.profile
                                # Ensure type is >= int16

    template['dtype'] = data_type
    template['nodata'] = nodata

    # Evaluate cloud cover and efficacy if band is quality
    efficacy = 0
    cloudratio = 100
    raster = None
    if band == quality_band:
        raster_merge, efficacy, cloudratio = getMask(raster_merge, datasets, mask=mask, compute=compute)

    # Ensure file tree is created
    merge_file = Path(merge_file)
    merge_file.parent.mkdir(parents=True, exist_ok=True)

    template.update({
        'tiled': True,
        "interleave": "pixel",
    })

    options = dict(
        file=str(merge_file),
        efficacy=efficacy,
        cloudratio=cloudratio,
        dataset=dataset,
        resolution=resx,
        nodata=nodata
    )

    if band == quality_band and len(datasets) > 1:
        provenance = merge_file.parent / merge_file.name.replace(band, DATASOURCE_NAME)

        profile = deepcopy(template)
        profile['dtype'] = DATASOURCE_ATTRIBUTES['data_type']
        profile['nodata'] = DATASOURCE_ATTRIBUTES['nodata']

        custom_tags = {dataset: value for value, dataset in enumerate(datasets)}

        save_as_cog(str(provenance), raster_provenance, tags=custom_tags, block_size=block_size, **profile)
        options[DATASOURCE_NAME] = str(provenance)

    # Persist on file as Cloud Optimized GeoTIFF
    save_as_cog(str(merge_file), raster_merge.astype(data_type), block_size=block_size, **template)

    return options


def _check_rio_file_access(url: str, access_token: str = None):
    """Make a HEAD request in order to check if the given resource is available and reachable."""
    headers = dict()
    if access_token:
        headers.update({'X-Api-Key': access_token})
    try:
        if url and not url.startswith('http'):
            return

        _ = requests.head(url, headers=headers)
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f'Connection refused {e.request.url}')
    except requests.exceptions.HTTPError as e:
        if e.response is None:
            raise
        reason = e.response.reason
        msg = str(e)
        if e.response.status_code == 403:
            if e.request.headers.get('x-api-key') or 'access_token=' in e.request.url:
                msg = "You don't have permission to request this resource."
            else:
                msg = 'Missing Authentication Token.'
        elif e.response.status_code == 500:
            msg = 'Could not request this resource.'

        raise requests.exceptions.HTTPError(f'({reason}) {msg}', request=e.request, response=e.response)


def post_processing_quality(quality_file: str, bands: List[str], cube: str,
                            date: str, tile_id, quality_band: str, band_map: dict, version: int, block_size:int=None):
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
         cube: Identity data cube name
         date: Merge date
         tile_id: Brazil data cube tile identifier
         quality_band: Quality band name
         version: Data cube version
    """
    # Get quality profile and chunks
    with rasterio.open(str(quality_file)) as merge_dataset:
        blocks = list(merge_dataset.block_windows())
        profile = merge_dataset.profile
        nodata = profile.get('nodata', band_map[quality_band]['nodata'])
        if nodata is not None:
            nodata = float(nodata)
        raster_merge = merge_dataset.read(1)

    _default_bands = DATASOURCE_NAME, 'ndvi', 'evi', 'cnc', TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME

    bands_without_quality = [b for b in bands if b != quality_band and b.lower() not in _default_bands]

    for _, block in blocks:
        nodata_positions = []

        row_offset = block.row_off + block.height
        col_offset = block.col_off + block.width

        for band in bands_without_quality:
            band_file = build_cube_path(get_cube_id(cube), date, tile_id, version=version, band=band)

            with rasterio.open(str(band_file)) as ds:
                raster = ds.read(1, window=block)

            band_nodata = band_map[band]['nodata']
            nodata_found = numpy.where(raster == float(band_nodata))
            raster_nodata_pos = numpy.ravel_multi_index(nodata_found, raster.shape)
            nodata_positions = numpy.union1d(nodata_positions, raster_nodata_pos)

        if len(nodata_positions):
            raster_merge[block.row_off: row_offset, block.col_off: col_offset][
                numpy.unravel_index(nodata_positions.astype(numpy.int64), raster.shape)] = nodata

    save_as_cog(str(quality_file), raster_merge, block_size=block_size, **profile)


class SmartDataSet:
    """Defines utility class to auto close rasterio data set.

    This class is class helper to avoid memory leak of opened data set in memory.
    """

    def __init__(self, file_path: str, mode='r', tags=None, **properties):
        """Initialize SmartDataSet definition and open rasterio data set."""
        self.path = Path(file_path)
        self.mode = mode
        self.dataset = rasterio.open(file_path, mode=mode, **properties)
        self.tags = tags
        self.mode = mode

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
            logging.debug('Closing dataset {}'.format(str(self.path)))

            if self.mode == 'w' and self.tags:
                self.dataset.update_tags(**self.tags)

            self.dataset.close()


def compute_data_set_stats(file_path: str, mask: dict, compute: bool = True) -> Tuple[float, float]:
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

        efficacy, cloud_ratio = _qa_statistics(raster, mask=mask, compute=compute)

    return efficacy, cloud_ratio


def blend(activity, band_map, quality_band, build_clear_observation=False, block_size=None):
    """Apply blend and generate raster from activity.

    Basically, the blend operation consists in stack all the images (merges) in period. The stack is based in
    best pixel image (Best clear ratio). The cloud pixels are masked with `numpy.ma` module, enabling to apply
    temporal composite function MEDIAN, AVG over these rasters.

    The following example represents a data cube Landsat-8 16 days using function Best Pixel (Stack - STK) and
    Median (MED) in period of 16 days from 1/1 to 16/1. The images from `10/1` and `15/1` were found and the values as
    described below::

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
    The result data cube will be::

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

    Note:
        When build_clear_observation is set, make sure to do not execute in parallel processing
        since it is not `thread-safe`.
        The provenance band is not generated by MEDIAN products.
        For pixels `nodata` in the best image, the cube builder will try to find useful pixel in the next observation.
        It may be cloud/cloud-shadow (when there is no valid pixel 0 and 1). Otherwise, fill as `nodata`.

    See Also:
        Numpy Masked Arrays https://numpy.org/doc/stable/reference/maskedarray.generic.html
        Brazil Data Cube Temporal Compositing https://brazil-data-cube.github.io/products/specifications/processing-flow.html#temporal-compositing

    Args:
        activity: Prepared blend activity metadata
        band_map: Map of data cube bands (common_name : name)
        build_clear_observation: Flag to dispatch generation of Clear Observation band. It is not ``thread-safe``.

    Returns:
        A processed activity with the generated values.
    """
    from .image import radsat_extract_bits

    # Assume that it contains a band and quality band
    numscenes = len(activity['scenes'])

    band = activity['band']
    activity_mask = activity['mask']
    mask_values = None

    version = activity['version']

    nodata = activity.get('nodata', band_map[band]['nodata'])
    if band == quality_band:
        nodata = activity_mask['nodata']

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

        efficacy = scene['efficacy']
        resolution = resolution
        mask_tuples.append((100. * efficacy / resolution, key))

    # Open all input files and save the datasets in two lists, one for masks and other for the current band.
    # The list will be ordered by efficacy/resolution
    masklist = []
    bandlist = []

    provenance_merge_map = dict()

    for m in sorted(mask_tuples, reverse=True):
        key = m[1]
        efficacy = m[0]
        scene = activity['scenes'][key]

        filename = scene['ARDfiles'][quality_band]
        quality_ref = rasterio.open(filename)

        if mask_values is None:
            mask_values = parse_mask(activity_mask)

        try:
            masklist.append(quality_ref)
        except BaseException as e:
            raise IOError('FileError while opening {} - {}'.format(filename, e))

        filename = scene['ARDfiles'][band]

        provenance_merge_map.setdefault(key, None)

        if scene['ARDfiles'].get(DATASOURCE_NAME):
            provenance_merge_map[key] = SmartDataSet(scene['ARDfiles'][DATASOURCE_NAME])

        try:
            bandlist.append(rasterio.open(filename))
        except BaseException as e:
            raise IOError('FileError while opening {} - {}'.format(filename, e))

    # Build the raster to store the output images.
    width = profile['width']
    height = profile['height']

    # Get the map values
    clear_values = mask_values['clear_data']
    not_clear_values = mask_values['not_clear_data']
    saturated_list = []
    if mask_values.get('saturated_band'):
        for m in sorted(mask_tuples, reverse=True):
            key = m[1]
            scene = activity['scenes'][key]

            filename = scene['ARDfiles'][mask_values['saturated_band']]
            saturated_file = SmartDataSet(filename, mode='r')
            saturated_list.append(saturated_file)

    saturated_values = mask_values['saturated_data']

    # STACK will be generated in memory
    stack_raster = numpy.full((height, width), dtype=profile['dtype'], fill_value=nodata)
    # Build the stack total observation
    stack_total_observation = numpy.zeros((height, width), dtype=numpy.uint8)

    datacube = activity.get('datacube')
    period = activity.get('period')
    tile_id = activity.get('tile_id')

    is_combined_collection = len(activity['datasets']) > 1

    cube_file = build_cube_path(datacube, period, tile_id, version=version, band=band, suffix='.tif')

    # Create directory
    cube_file.parent.mkdir(parents=True, exist_ok=True)

    cube_function = DataCubeFragments(datacube).composite_function

    if cube_function == 'MED':
        median_raster = numpy.full((height, width), fill_value=nodata, dtype=profile['dtype'])

    if build_clear_observation:
        logging.warning('Creating and computing Clear Observation (ClearOb) file...')

        clear_ob_file_path = build_cube_path(datacube, period, tile_id, version=version, band=CLEAR_OBSERVATION_NAME, suffix='.tif')
        dataset_file_path = build_cube_path(datacube, period, tile_id, version=version, band=DATASOURCE_NAME, suffix='.tif')

        clear_ob_profile = profile.copy()
        clear_ob_profile['dtype'] = CLEAR_OBSERVATION_ATTRIBUTES['data_type']
        clear_ob_profile.pop('nodata', None)
        clear_ob_data_set = SmartDataSet(str(clear_ob_file_path), 'w', **clear_ob_profile)

        dataset_profile = profile.copy()
        dataset_profile['dtype'] = DATASOURCE_ATTRIBUTES['data_type']
        dataset_profile['nodata'] = DATASOURCE_ATTRIBUTES['nodata']

        if is_combined_collection:
            datasets = activity['datasets']
            tags = {dataset: value for value, dataset in enumerate(datasets)}

            datasource = SmartDataSet(str(dataset_file_path), 'w', tags=tags, **dataset_profile)
            datasource.dataset.write(numpy.full((height, width),
                                              fill_value=DATASOURCE_ATTRIBUTES['nodata'],
                                              dtype=DATASOURCE_ATTRIBUTES['data_type']), indexes=1)

    provenance_array = numpy.full((height, width), dtype=numpy.int16, fill_value=-1)

    for _, window in tilelist:
        # Build the stack to store all images as a masked array. At this stage the array will contain the masked data
        stackMA = numpy.ma.zeros((numscenes, window.height, window.width), dtype=numpy.int16)

        notdonemask = numpy.ones(shape=(window.height, window.width), dtype=numpy.bool_)

        if build_clear_observation and is_combined_collection:
            data_set_block = numpy.full((window.height, window.width),
                                        fill_value=DATASOURCE_ATTRIBUTES['nodata'],
                                        dtype=DATASOURCE_ATTRIBUTES['data_type'])

        row_offset = window.row_off + window.height
        col_offset = window.col_off + window.width

        # For all pair (quality,band) scenes
        for order in range(numscenes):
            # Read both chunk of Merge and Quality, respectively.
            ssrc = bandlist[order]
            msrc = masklist[order]
            raster = ssrc.read(1, window=window)
            masked = msrc.read(1, window=window, masked=True)
            copy_mask = numpy.array(masked, copy=True)

            if saturated_list:
                saturated = saturated_list[order].dataset.read(1, window=window)
                # TODO: Get the original band order and apply to the extract function instead.
                saturated = radsat_extract_bits(saturated, 1, 7).astype(numpy.bool_)

                masked.mask[saturated] = True

            # Mask cloud/snow/shadow/no-data as False
            masked.mask[numpy.where(numpy.isin(masked, not_clear_values))] = True
            # Ensure that Raster noda value (-9999 maybe) is set to False
            masked.mask[raster == nodata] = True
            masked.mask[numpy.where(numpy.isin(masked, saturated_values))] = True
            # Mask valid data (0 and 1) as True
            masked.mask[numpy.where(numpy.isin(masked, clear_values))] = False

            # Create an inverse mask value in order to pass to numpy masked array
            # True => nodata
            bmask = masked.mask

            # Use the mask to mark the fill (0) and cloudy (2) pixels
            stackMA[order] = numpy.ma.masked_where(bmask, raster)

            # Copy Masked values in order to stack total observation
            copy_mask[copy_mask != nodata] = 1
            copy_mask[copy_mask == nodata] = 0

            stack_total_observation[window.row_off: row_offset, window.col_off: col_offset] += copy_mask.astype(numpy.uint8)

            # Get current observation file name
            file_name = Path(bandlist[order].name).stem
            file_date = datetime.strptime(file_name.split('_')[4], '%Y-%m-%d')
            day_of_year = file_date.timetuple().tm_yday

            # Find all no data in destination STACK image
            stack_raster_where_nodata = numpy.where(
                stack_raster[window.row_off: row_offset, window.col_off: col_offset] == nodata
            )

            # Turns into a 1-dimension
            stack_raster_nodata_pos = numpy.ravel_multi_index(stack_raster_where_nodata,
                                                              stack_raster[window.row_off: row_offset,
                                                              window.col_off: col_offset].shape)

            if build_clear_observation and is_combined_collection:
                datasource_block = provenance_merge_map[file_date.strftime('%Y-%m-%d')].dataset.read(1, window=window)

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

                if build_clear_observation and is_combined_collection:
                    data_set_block[where_intersec] = datasource_block[where_intersec]

            # Identify what is needed to stack, based in Array 2d bool
            todomask = notdonemask * numpy.invert(bmask)

            # Find all positions where valid data matches.
            clear_not_done_pixels = numpy.where(numpy.logical_and(todomask, numpy.invert(masked.mask)))

            # Override the STACK Raster with valid data.
            stack_raster[window.row_off: row_offset, window.col_off: col_offset][clear_not_done_pixels] = raster[
                clear_not_done_pixels]

            # Mark day of year to the valid pixels
            provenance_array[window.row_off: row_offset, window.col_off: col_offset][
                clear_not_done_pixels] = day_of_year

            if build_clear_observation and is_combined_collection:
                data_set_block[clear_not_done_pixels] = datasource_block[clear_not_done_pixels]

            # Update what was done.
            notdonemask = notdonemask * bmask

        if cube_function == 'MED':
            median = numpy.ma.median(stackMA, axis=0).data
            median[notdonemask.astype(numpy.bool_)] = nodata

            median_raster[window.row_off: row_offset, window.col_off: col_offset] = median.astype(profile['dtype'])

        if build_clear_observation:
            count_raster = numpy.ma.count(stackMA, axis=0)

            clear_ob_data_set.dataset.write(count_raster.astype(clear_ob_profile['dtype']), window=window, indexes=1)

            if is_combined_collection:
                datasource.dataset.write(data_set_block, window=window, indexes=1)

    # Close all input dataset
    for order in range(numscenes):
        bandlist[order].close()
        masklist[order].close()

    # Evaluate cloud cover
    efficacy, cloudcover = _qa_statistics(stack_raster, mask=mask_values, compute=False)

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

        total_observation_file = build_cube_path(datacube, period, tile_id, version=version, band=TOTAL_OBSERVATION_NAME)
        total_observation_profile = profile.copy()
        total_observation_profile.pop('nodata', None)
        total_observation_profile['dtype'] = 'uint8'

        save_as_cog(str(total_observation_file), stack_total_observation, block_size=block_size, **total_observation_profile)
        generate_cogs(str(clear_ob_file_path), str(clear_ob_file_path), block_size=block_size)

        activity['clear_observation_file'] = str(clear_ob_data_set.path)
        activity['total_observation'] = str(total_observation_file)

    if cube_function == 'MED':
        # Close and upload the MEDIAN dataset
        save_as_cog(str(cube_file), median_raster, block_size=block_size, mode='w', **profile)
    else:
        save_as_cog(str(cube_file), stack_raster, block_size=block_size, mode='w', **profile)

        if build_clear_observation:
            provenance_file = build_cube_path(datacube, period, tile_id, version=version, band=PROVENANCE_NAME)
            provenance_profile = profile.copy()
            provenance_profile.pop('nodata', -1)
            provenance_profile['dtype'] = PROVENANCE_ATTRIBUTES['data_type']

            save_as_cog(str(provenance_file), provenance_array, block_size=block_size, **provenance_profile)
            activity['provenance'] = str(provenance_file)

            if is_combined_collection:
                datasource.close()
                generate_cogs(str(dataset_file_path), str(dataset_file_path), block_size=block_size)
                activity['datasource'] = str(dataset_file_path)

    activity['blends'] = {
        cube_function: str(cube_file)
    }

    # Release reference count
    stack_raster = None

    activity['efficacy'] = efficacy
    activity['cloudratio'] = cloudcover

    return activity


def generate_rgb(rgb_file: Path, qlfiles: List[str]):
    """Generate a raster file that stack the quick look files into RGB channel."""
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


def concat_path(*entries) -> Path:
    """Concat any path and retrieves a pathlib.Path.

    Note:
        This method resolves the path concatenation when right argument starts with slash /.
        The default python join does not merge any right path when starts with slash.

    Examples:
        >>> print(str(concat_path('/path', '/any/path/')))
        ... '/path/any/path/'
    """
    base = Path('/')

    for entry in entries:
        base /= entry if not str(entry).startswith('/') else str(entry)[1:]

    return base


def _item_prefix(absolute_path: Path) -> Path:
    relative_path = Path(absolute_path).relative_to(Config.DATA_DIR)
    relative_path = relative_path.relative_to('Repository')

    return concat_path(Config.ITEM_PREFIX, relative_path)


def publish_datacube(cube, bands, tile_id, period, scenes, cloudratio, band_map, **kwargs):
    """Generate quicklook and catalog datacube on database."""
    start_date, end_date = period.split('_')

    datacube = cube.name

    cube_parts = get_cube_parts(datacube)

    for composite_function in [cube_parts.composite_function]:
        item_datacube = get_cube_id(datacube, composite_function)

        item_id = get_item_id(item_datacube, cube.version, tile_id, period)

        cube_bands = cube.bands

        quick_look_file = build_cube_path(item_datacube, period, tile_id, version=cube.version, suffix=None)

        ql_files = []
        for band in bands:
            ql_files.append(scenes[band][composite_function])

        quick_look_file = generate_quick_look(str(quick_look_file), ql_files)

        if kwargs.get('with_rgb'):
            rgb_file = build_cube_path(item_datacube, period, tile_id, version=cube.version, band='RGB')
            generate_rgb(rgb_file, ql_files)

        map_band_scene = {name: composite_map[composite_function] for name, composite_map in scenes.items()}

        custom_bands = generate_band_indexes(cube, map_band_scene, period, tile_id)
        for name, file in custom_bands.items():
            scenes[name] = {composite_function: str(file)}

        tile = Tile.query().filter(Tile.name == tile_id, Tile.grid_ref_sys_id == cube.grid_ref_sys_id).first()

        with db.session.begin_nested():
            item_data = dict(
                name=item_id,
                collection_id=cube.id,
                tile_id=tile.id,
                start_date=start_date,
                end_date=end_date,
            )

            item, _ = get_or_create_model(Item, defaults=item_data, name=item_id, collection_id=cube.id)
            item.cloud_cover = cloudratio

            assets = deepcopy(item.assets) or dict()
            assets.update(
                thumbnail=create_asset_definition(
                    href=str(_item_prefix(Path(quick_look_file))),
                    mime_type=PNG_MIME_TYPE,
                    role=['thumbnail'],
                    absolute_path=str(quick_look_file)
                )
            )
            item.start_date = start_date
            item.end_date = end_date

            extent = to_shape(item.geom) if item.geom else None
            min_convex_hull = to_shape(item.min_convex_hull) if item.min_convex_hull else None

            for band in scenes:
                band_model = list(filter(lambda b: b.name == band, cube_bands))

                # Band does not exists on model
                if not band_model:
                    logging.warning('Band {} of {} does not exist on database. Skipping'.format(band, cube.id))
                    continue

                if extent is None:
                    extent = raster_extent(str(scenes[band][composite_function]))

                if min_convex_hull is None:
                    min_convex_hull = raster_convexhull(str(scenes[band][composite_function]))

                assets[band_model[0].name] = create_asset_definition(
                    href=str(_item_prefix(scenes[band][composite_function])),
                    mime_type=COG_MIME_TYPE,
                    role=['data'],
                    absolute_path=str(scenes[band][composite_function]),
                    is_raster=True
                )

            item.assets = assets
            item.srid = SRID_ALBERS_EQUAL_AREA
            if min_convex_hull.area > 0.0:
                item.min_convex_hull = from_shape(min_convex_hull, srid=4326)
            item.geom = from_shape(extent, srid=4326)

        db.session.commit()

    return quick_look_file


def publish_merge(bands, datacube, tile_id, date, scenes, band_map):
    """Generate quicklook and catalog warped datacube on database.

    TODO: Review it with publish_datacube
    """
    item_id = get_item_id(datacube.name, datacube.version, tile_id, date)

    quick_look_file = build_cube_path(datacube.name, date, tile_id, version=datacube.version, suffix=None)

    cube_bands = datacube.bands

    ql_files = []
    for band in bands:
        ql_files.append(scenes['ARDfiles'][band])

    quick_look_file = generate_quick_look(str(quick_look_file), ql_files)

    # Generate VI
    custom_bands = generate_band_indexes(datacube, scenes['ARDfiles'], date, tile_id)
    scenes['ARDfiles'].update(custom_bands)

    tile = Tile.query().filter(Tile.name == tile_id, Tile.grid_ref_sys_id == datacube.grid_ref_sys_id).first()

    with db.session.begin_nested():
        item_data = dict(
            name=item_id,
            collection_id=datacube.id,
            tile_id=tile.id,
            start_date=date,
            end_date=date,
        )

        item, _ = get_or_create_model(Item, defaults=item_data, name=item_id, collection_id=datacube.id)
        item.cloud_cover = scenes.get('cloudratio', 0)

        extent = to_shape(item.geom) if item.geom else None
        min_convex_hull = to_shape(item.min_convex_hull) if item.min_convex_hull else None

        assets = deepcopy(item.assets) or dict()

        assets.update(
            thumbnail=create_asset_definition(
                href=str(_item_prefix(Path(quick_look_file))),
                mime_type=PNG_MIME_TYPE,
                role=['thumbnail'],
                absolute_path=str(quick_look_file)
            )
        )
        item.start_date = date
        item.end_date = date

        for band in scenes['ARDfiles']:
            band_model = list(filter(lambda b: b.name == band, cube_bands))

            # Band does not exists on model
            if not band_model:
                logging.warning('Band {} of {} does not exist on database'.format(band, datacube.id))
                continue

            if extent is None:
                extent = raster_extent(str(scenes['ARDfiles'][band]))

            if min_convex_hull is None:
                min_convex_hull = raster_convexhull(str(scenes['ARDfiles'][band]))

            assets[band_model[0].name] = create_asset_definition(
                href=str(_item_prefix(scenes['ARDfiles'][band])),
                mime_type=COG_MIME_TYPE,
                role=['data'],
                absolute_path=str(scenes['ARDfiles'][band]),
                is_raster=True
            )

        item.srid = SRID_ALBERS_EQUAL_AREA
        item.geom = from_shape(extent, srid=4326)
        if min_convex_hull.area > 0.0:
            item.min_convex_hull = from_shape(min_convex_hull, srid=4326)
        item.assets = assets

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


def getMask(raster, dataset=None, mask=None, compute=False):
    """Retrieve and re-sample quality raster to well-known values used in Brazil Data Cube.

    Args:
        raster - Raster with Quality band values
        satellite - Satellite Type

    Returns:
        Tuple containing formatted quality raster, efficacy and cloud ratio, respectively.
    """
    rastercm = raster

    efficacy, cloudratio = _qa_statistics(rastercm, mask=mask, compute=compute)

    return rastercm.astype(numpy.uint8), efficacy, cloudratio


def parse_mask(mask: dict):
    """Parse input mask according to the raster.

    This method expects a dict with contains the following keys:

        clear_data - Array of clear data
        nodata - Cloud mask nodata
        not_clear (Optional) _ List of pixels to be not considered.

    It will read the input array and get all unique values. Make sure to call for cloud file.

    The following section describe an example how to pass the mask values for any cloud processor:

        fmask = dict(
            clear_data=[0, 1],
            not_clear_data=[2, 3, 4],
            nodata=255
        )

        sen2cor = dict(
            clear_data=[4, 5, 6, 7],
            not_clear_data=[2, 3, 8, 9, 10, 11],
            saturated_data=[1],
            nodata=0
        )

    Note:
        It may take too long do parse, according to the raster.
        Not mapped values will be treated as "others" and may not be count in the Clear Observation Band (CLEAROB).

    Args:
        mask (dict): Mapping for cloud masking values.

    Returns:
        dict Representing the mapped values of cloud mask.
    """
    clear_data = numpy.array(mask['clear_data'])
    not_clear_data = numpy.array(mask.get('not_clear_data', []))
    saturated_data = mask.get('saturated_data', [])

    if mask.get('nodata') is None:
        raise RuntimeError('Expected nodata value set to compute data set statistics.')

    nodata = mask['nodata']

    res = dict(
        clear_data=clear_data,
        not_clear_data=not_clear_data,
        saturated_data=saturated_data,
        nodata=nodata,
    )

    if mask.get('saturated_band'):
        res['saturated_band'] = mask['saturated_band']

    return res


def _qa_statistics(raster, mask: dict, compute: bool = False) -> Tuple[float, float]:
    """Retrieve raster statistics efficacy and not clear ratio, based in Fmask values.

    Note:
        Values 0 and 1 are considered `clear data`.
        Values 2 and 4 are considered as `not clear data`
        The values for snow `3` and nodata `255` is not used to count efficacy and not clear ratio
    """
    if compute:
        mask = parse_mask(mask)

    # Compute how much data is for each class. It will be used as image area
    clear_pixels = raster[numpy.where(numpy.isin(raster, mask['clear_data']))].size
    not_clear_pixels = raster[numpy.where(numpy.isin(raster, mask['not_clear_data']))].size

    # Total pixels used to retrieve data efficacy
    total_pixels = raster.size
    # Image area is everything, except nodata.
    image_area = clear_pixels + not_clear_pixels
    not_clear_ratio = 100

    if image_area != 0:
        not_clear_ratio = round(100. * not_clear_pixels / image_area, 2)

    efficacy = round(100. * clear_pixels / total_pixels, 2)

    return efficacy, not_clear_ratio


def build_cube_path(datacube: str, period: str, tile_id: str, version: int, band: str = None, suffix: Union[str, None] = '.tif') -> Path:
    """Retrieve the path to the Data cube file in Brazil Data Cube Cluster."""
    folder = 'Warped'
    date = period

    fragments = DataCubeFragments(datacube)

    version_str = 'v{0:03d}'.format(version)

    file_name = f'{datacube}_{version_str}_{tile_id}_{period}'

    if band is not None:
        file_name = f'{file_name}_{band}'

    if suffix is None:
        suffix = ''

    file_name = f'{file_name}{suffix}'

    if fragments.composite_function != 'IDT':  # For cube with temporal composition
        folder = 'Mosaic'

    return Path(Config.DATA_DIR) / 'Repository' / folder / datacube / version_str / tile_id / date / file_name


def create_asset_definition(href: str, mime_type: str, role: List[str], absolute_path: str,
                            created=None, is_raster=False):
    """Create a valid asset definition for collections.

    TODO: Generate the asset for `Item` field with all bands

    Args:
        href - Relative path to the asset
        mime_type - Asset Mime type str
        role - Asset role. Available values are: ['data'], ['thumbnail']
        absolute_path - Absolute path to the asset. Required to generate check_sum
        created - Date time str of asset. When not set, use current timestamp.
        is_raster - Flag to identify raster. When set, `raster_size` and `chunk_size` will be set to the asset.
    """
    fmt = '%Y-%m-%dT%H:%M:%S'
    _now_str = datetime.utcnow().strftime(fmt)

    if created is None:
        created = _now_str
    elif isinstance(created, datetime):
        created = created.strftime(fmt)

    asset = {
        'href': str(href),
        'type': mime_type,
        'bdc:size': Path(absolute_path).stat().st_size,
        'checksum:multihash': multihash_checksum_sha256(str(absolute_path)),
        'roles': role,
        'created': created,
        'updated': _now_str
    }

    if is_raster:
        with rasterio.open(str(absolute_path)) as data_set:
            asset['bdc:raster_size'] = dict(
                x=data_set.shape[1],
                y=data_set.shape[0],
            )

            chunk_x, chunk_y = data_set.profile.get('blockxsize'), data_set.profile.get('blockxsize')

            if chunk_x is None or chunk_x is None:
                return asset

            asset['bdc:chunk_size'] = dict(x=chunk_x, y=chunk_y)

    return asset


def save_as_cog(destination: str, raster, mode='w', tags=None, block_size=None, **profile):
    """Save the raster file as Cloud Optimized GeoTIFF.

    See Also:
        Cloud Optimized GeoTiff https://gdal.org/drivers/raster/cog.html

    Args:
        destination: Path to store the data set.
        raster: Numpy raster values to persist in disk
        mode: Default rasterio mode. Default is 'w' but you also can set 'r+'.
        tags: Tag values (Dict[str, str]) to write on dataset.
        **profile: Rasterio profile values to add in dataset.
    """
    with rasterio.open(str(destination), mode, **profile) as dataset:
        if profile.get('nodata'):
            dataset.nodata = profile['nodata']

        dataset.write_band(1, raster)

        if tags:
            dataset.update_tags(**tags)

    generate_cogs(str(destination), str(destination), block_size=block_size)


def generate_cogs(input_data_set_path, file_path, profile='deflate', block_size=None, profile_options=None, **options):
    """Generate Cloud Optimized GeoTIFF files (COG).

    Args:
        input_data_set_path (str) - Path to the input data set
        file_path (str) - Target data set filename
        profile (str) - A COG profile based in `rio_cogeo.profiles`.
        profile_options (dict) - Custom options to the profile.
        block_size (int) - Custom block size.

    Returns:
        Path to COG.
    """
    if profile_options is None:
        profile_options = dict()

    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)

    if block_size:
        output_profile["blockxsize"] = block_size
        output_profile["blockysize"] = block_size

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )

    cog_translate(
        str(input_data_set_path),
        str(file_path),
        output_profile,
        config=config,
        in_memory=False,
        quiet=True,
        **options,
    )
    return str(file_path)


@contextmanager
def rasterio_access_token(access_token=None):
    """Retrieve a context manager that wraps a temporary file containing the access token to be passed to STAC."""
    with TemporaryDirectory() as tmp:
        options = dict()

        if access_token:
            tmp_file = Path(tmp) / f'{access_token}.txt'
            with open(str(tmp_file), 'w') as f:
                f.write(f'X-Api-Key: {access_token}')
            options.update(GDAL_HTTP_HEADER_FILE=str(tmp_file))

        yield options


def raster_convexhull(imagepath: str, epsg='EPSG:4326') -> dict:
    """Get a raster image footprint.

    Args:
        imagepath (str): image file
        epsg (str): geometry EPSG

    See:
        https://rasterio.readthedocs.io/en/latest/topics/masks.html
    """
    with rasterio.open(imagepath) as dataset:
        # Read raster data, masking nodata values
        data = dataset.read(1, masked=True)
        # Create mask, which 1 represents valid data and 0 nodata
        mask = numpy.invert(data.mask).astype(numpy.uint8)

        geoms = []
        res = {'val': []}
        for geom, val in rasterio.features.shapes(mask, mask=mask, transform=dataset.transform):
            geom = rasterio.warp.transform_geom(dataset.crs, epsg, geom, precision=6)

            res['val'].append(val)
            geoms.append(shapely.geometry.shape(geom))

        if len(geoms) == 1:
            return geoms[0]

        multi_polygons = shapely.geometry.MultiPolygon(geoms)

        return multi_polygons.convex_hull


def raster_extent(imagepath: str, epsg = 'EPSG:4326') -> shapely.geometry.Polygon:
    """Get raster extent in arbitrary CRS.

    Args:
        imagepath (str): Path to image
        epsg (str): EPSG Code of result crs
    Returns:
        dict: geojson-like geometry
    """
    with rasterio.open(imagepath) as dataset:
        _geom = shapely.geometry.mapping(shapely.geometry.box(*dataset.bounds))
        return shapely.geometry.shape(rasterio.warp.transform_geom(dataset.crs, epsg, _geom, precision=6))
