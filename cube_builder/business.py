#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder business interface."""

from typing import Tuple

# 3rdparty
from bdc_db.models import Band, Collection, TemporalCompositionSchema
from bdc_db.models.base_sql import BaseModel
from werkzeug.exceptions import Conflict, NotFound

from .forms import CollectionForm, TemporalSchemaForm
from .image import validate_merges
from .maestro import Maestro, decode_periods
from .models import Activity
from .utils import get_cube_id, get_cube_parts, get_or_create_model


class CubeBusiness:
    """Define Cube Builder interface for data cube creation."""

    @classmethod
    def create(cls, params: dict):
        """Create and persist datacube on database."""
        params['composite_function_list'] = ['IDENTITY', 'STK', 'MED']

        # generate cubes metadata
        cubes_db = Collection.query().filter().all()
        cubes = []
        cubes_serealized = []

        for composite_function in params['composite_function_list']:
            c_function_id = composite_function.upper()

            cube_id = get_cube_id(params['datacube'], c_function_id)

            raster_size_id = '{}-{}'.format(params['grs'], int(params['resolution']))

            temporal_composition = params['temporal_schema'] if c_function_id.upper() != 'IDENTITY' else 'Anull'

            # add cube
            if not list(filter(lambda x: x.id == cube_id, cubes)) and not list(filter(lambda x: x.id == cube_id, cubes_db)):
                cube = Collection(
                    id=cube_id,
                    temporal_composition_schema_id=temporal_composition,
                    raster_size_schema_id=raster_size_id,
                    composite_function_schema_id=c_function_id,
                    grs_schema_id=params['grs'],
                    description=params['description'],
                    radiometric_processing=None,
                    geometry_processing=None,
                    sensor=None,
                    is_cube=True,
                    oauth_scope=params.get('oauth_scope', None),
                    bands_quicklook=','.join(params['bands_quicklook']),
                    license=params.get('license')
                )

                cubes.append(cube)
                cubes_serealized.append(CollectionForm().dump(cube))

        BaseModel.save_all(cubes)

        bands = []

        for cube in cubes:
            fragments = get_cube_parts(cube.id)

            # A IDENTITY data cube is composed by CollectionName and Resolution (LC8_30, S2_10)
            is_identity = len(fragments) == 2

            # save bands
            for band in params['bands']:
                # Skip creation of band CNC for IDENTITY data cube
                # or band quality for composite data cube
                if (band == 'cnc' and is_identity) or (band == 'quality' and not is_identity):
                    continue

                is_not_cloud = band != 'quality' and band != 'cnc'

                band = band.strip()
                bands.append(Band(
                    name=band,
                    collection_id=cube.id,
                    min=0 if is_not_cloud else 0,
                    max=10000 if is_not_cloud else 255,
                    fill=-9999 if is_not_cloud else 0,
                    scale=0.0001 if is_not_cloud else 1,
                    data_type='int16' if is_not_cloud else 'Uint16',
                    common_name=band,
                    resolution_x=params['resolution'],
                    resolution_y=params['resolution'],
                    resolution_unit='m',
                    description='',
                    mime_type='image/tiff'
                ))

        BaseModel.save_all(bands)

        return cubes_serealized, 201

    @classmethod
    def maestro(cls, datacube, collections, tiles, start_date, end_date, **properties):
        """Search and Dispatch datacube generation on cluster.

        Args:
            datacube - Data cube name.
            collections - List of collections used to generate datacube.
            tiles - List of tiles to generate.
            start_date - Start period
            end_date - End period
            **properties - Additional properties used on datacube generation, such bands and cache.
        """
        maestro = Maestro(datacube, collections, tiles, start_date, end_date, **properties)

        maestro.orchestrate()

        maestro.dispatch_celery()

        return dict(ok=True)

    @classmethod
    def check_for_invalid_merges(cls, datacube: str, tile: str, start_date: str, last_date: str) -> dict:
        """List merge files used in data cube and check for invalid scenes.

        Args:
            datacube: Data cube name
            tile: Brazil Data Cube Tile identifier
            start_date: Activity start date (period)
            last_date: Activity End (period)

        Returns:
            List of Images used in period
        """
        cube = Collection.query().filter(Collection.id == datacube).first()

        if cube is None or not cube.is_cube:
            raise NotFound('Cube {} not found'.format(datacube))

        # TODO validate schema to avoid start/end too abroad

        res = Activity.list_merge_files(datacube, tile, start_date, last_date)

        result = validate_merges(res)

        return result, 200

    @classmethod
    def create_temporal_composition(cls, params: dict) -> Tuple[dict, int]:
        """Create a temporal composition schema on database.

        The TemporalCompositionSchema is used to describe how the data cube will be created.

        You can define a data cube montly, each 16 days, season, etc. Once defined,
        the ``cube_builder`` will seek for all images within period given and will
        generate data cube passing these images to a composite function.

        Raises:
            Conflict when a duplicated composition is given.

        Args:
            params - Required parameters for a ``TemporalCompositionSchema``.

        Returns:
            Tuple with object created and respective HTTP Status code
        """
        object_id = '{}{}{}'.format(params['temporal_schema'],
                                    params['temporal_composite_t'],
                                    params['temporal_composite_unit'])

        temporal_schema, created = get_or_create_model(TemporalCompositionSchema, defaults=params, id=object_id)

        if created:
            # Persist
            temporal_schema.save()

            return TemporalSchemaForm().dump(temporal_schema), 201

        raise Conflict('Schema "{}" already exists.'.format(object_id))

    @classmethod
    def generate_periods(cls, schema, step, start_date=None, last_date=None, **kwargs) -> Tuple[str]:
        """Generate data cube periods using temporal composition schema.

        Args:
            schema: Temporal Schema (M, A)
            step: Temporal Step
            start_date: Start date offset. Default is '2016-01-01'.
            last_date: End data offset. Default is '2019-12-31'
            **kwargs: Optional parameters

        Returns:
            List of periods between start/last date
        """
        start_date = start_date or '2016-01-01'
        last_date = last_date or '2019-12-31'

        total_periods = decode_periods(schema, start_date, last_date, int(step))

        periods = set()

        for period_array in total_periods.values():
            for period in period_array:
                date = period.split('_')[0]

                periods.add(date)

        return sorted(list(periods))
