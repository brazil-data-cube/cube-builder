#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# 3rdparty
from bdc_db.models.base_sql import BaseModel
from bdc_db.models import Band, Collection

from .forms import CollectionForm


class CubeBusiness:
    @classmethod
    def create(cls, params: dict):
        params['composite_function_list'] = ['WARPED', 'STK', 'MED']

        # generate cubes metadata
        cubes_db = Collection.query().filter().all()
        cubes = []
        cubes_serealized = []

        for composite_function in params['composite_function_list']:
            c_function_id = composite_function.upper()
            cube_id = '{}_{}'.format(params['datacube'], c_function_id)

            raster_size_id = '{}-{}'.format(params['grs'], int(params['resolution']))

            # add cube
            if not list(filter(lambda x: x.id == cube_id, cubes)) and not list(filter(lambda x: x.id == cube_id, cubes_db)):
                cube = Collection(
                    id=cube_id,
                    temporal_composition_schema_id=params['temporal_schema'],
                    raster_size_schema_id=raster_size_id,
                    composite_function_schema_id=c_function_id,
                    grs_schema_id=params['grs'],
                    description=params['description'],
                    radiometric_processing=None,
                    geometry_processing=None,
                    sensor=None,
                    is_cube=True,
                    oauth_scope=None,
                    bands_quicklook=','.join(params['bands_quicklook'])
                )

                cubes.append(cube)
                cubes_serealized.append(CollectionForm().dump(cube))

        BaseModel.save_all(cubes)

        bands = []

        for cube in cubes:
            # save bands
            for band in params['bands']:
                band = band.strip()
                bands.append(Band(
                    name=band,
                    collection_id=cube.id,
                    min=0 if band != 'quality' else 0,
                    max=10000 if band != 'quality' else 255,
                    fill=-9999 if band != 'quality' else 0,
                    scale=0.0001 if band != 'quality' else 1,
                    data_type='int16' if band != 'quality' else 'Uint16',
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
    def maestro(cls, datacube, collections, tiles, start_date, end_date, bands=None):
        from .maestro import Maestro

        maestro = Maestro(datacube, collections, tiles, start_date, end_date, bands)

        maestro.orchestrate()

        maestro.dispatch_celery()

        return dict(ok=True)
