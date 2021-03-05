#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Simple data cube band generator."""

import logging
from typing import Dict, List

import numpy
from bdc_catalog.models import Band, Collection

from .interpreter import execute

BandMapFile = Dict[str, str]
"""Type which a key (represented as data cube band name) points to generated file in disk."""


def generate_band_indexes(cube: Collection, scenes: dict, period: str, tile_id: str) -> BandMapFile:
    """Generate data cube custom bands based in string-expression on table `band_indexes`.

    This method seeks for custom bands on Collection Band definition. A custom band must have
    `metadata` property filled out according the ``bdc_catalog.jsonschemas.band-metadata.json``.

    Notes:
        When collection does not have any index band, returns empty dict.

    Raises:
        RuntimeError when an error occurs while interpreting the band expression in Python Virtual Machine.

    Returns:
        A dict values with generated bands.
    """
    from .processing import SmartDataSet, build_cube_path, generate_cogs

    cube_band_indexes: List[Band] = []

    for band in cube.bands:
        if band._metadata and band._metadata.get('expression') and band._metadata['expression'].get('value'):
            cube_band_indexes.append(band)

    if not cube_band_indexes:
        return dict()

    map_data_set_context = dict()
    profile = None
    blocks = []

    for band_name, file_path in scenes.items():
        map_data_set_context[band_name] = SmartDataSet(str(file_path), mode='r')

        if profile is None:
            profile = map_data_set_context[band_name].dataset.profile.copy()
            blocks = list(map_data_set_context[band_name].dataset.block_windows())

    if not blocks or profile is None:
        raise RuntimeError('Can\t generate band indexes since profile/blocks is None.')

    output = dict()

    for band_index in cube_band_indexes:
        band_name = band_index.name

        band_expression = band_index._metadata['expression']['value']

        band_data_type = band_index.data_type

        data_type_info = numpy.iinfo(band_data_type)

        data_type_max_value = data_type_info.max
        data_type_min_value = data_type_info.min

        profile['dtype'] = band_data_type

        custom_band_path = build_cube_path(cube.name, period, tile_id, version=cube.version, band=band_name)

        output_dataset = SmartDataSet(str(custom_band_path), mode='w', **profile)
        logging.info(f'Generating band {band_name} for cube {cube.name} -{custom_band_path.stem}...')

        for _, window in blocks:
            machine_context = {
                k: ds.dataset.read(1, masked=True, window=window).astype(numpy.float32)
                for k, ds in map_data_set_context.items()
            }

            expr = f'{band_name} = {band_expression}'

            result = execute(expr, context=machine_context)
            raster = result[band_name]
            raster[raster == numpy.ma.masked] = profile['nodata']

            # Persist the expected band data type to cast value safely.
            # TODO: Should we use consider band min_value/max_value?
            raster[raster < data_type_min_value] = data_type_min_value
            raster[raster > data_type_max_value] = data_type_max_value

            output_dataset.dataset.write(raster.astype(band_data_type), window=window, indexes=1)

        output_dataset.close()

        generate_cogs(str(custom_band_path), str(custom_band_path))

        output[band_name] = str(custom_band_path)

    return output
