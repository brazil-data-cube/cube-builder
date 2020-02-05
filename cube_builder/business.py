#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# Python
from datetime import date as Date, datetime
from typing import List, Optional
# 3rdparty
from celery import chain, chord, group
from dateutil.relativedelta import relativedelta
from geoalchemy2 import func
from werkzeug.exceptions import NotAcceptable, NotFound

from bdc_db.models.base_sql import BaseModel
from bdc_db.models import Band, Collection, db, Tile
from .forms import CollectionForm


class CubeBusiness:
    @classmethod
    def create(cls, params: dict):
        # add WARPED type if not send
        if 'WARPED' not in [func.upper() for func in params['composite_function_list']]:
            params['composite_function_list'].append('WARPED')

        # generate cubes metadata
        cubes_db = Collection.query().filter().all()
        cubes = []
        cubes_serealized = []

        for composite_function in params['composite_function_list']:
            c_function_id = composite_function.upper()
            cube_id = '{}{}'.format(params['datacube'], c_function_id)

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
            for band in params['bands']['names']:
                band = band.strip()
                bands.append(Band(
                    name=band,
                    collection_id=cube.id,
                    min=params['bands']['min'],
                    max=params['bands']['max'],
                    fill=params['bands']['fill'],
                    scale=params['bands']['scale'],
                    data_type=params['bands']['data_type'],
                    common_name=band,
                    resolution_x=params['resolution'],
                    resolution_y=params['resolution'],
                    resolution_unit='m',
                    description='',
                    mime_type='image/tiff'
                ))

        BaseModel.save_all(bands)

        return cubes_serealized, 201

    @staticmethod
    def _prepare_blend_dates(cube: Collection, warp_merge: dict, start_date: Date, end_date: Date):
        requestedperiods = {}

        t_composite_schema = cube.temporal_composition_schema

        # tdtimestep = datetime.timedelta(days=int(t_composite_schema.temporal_composite_t))
        # stepsperperiod = int(round(365./timestep))

        # start_date = datetime.strptime('%Y-%m-%d', start_date).date()

        if end_date is None:
            end_date = datetime.now().date()

        if t_composite_schema.temporal_schema is None:
            periodkey = startdate + '_' + startdate + '_' + end_date
            requestedperiod = []
            requestedperiod.append(periodkey)
            requestedperiods[startdate] = requestedperiod
            return requestedperiods

        if t_composite_schema.temporal_schema == 'M':
            requestedperiod = []

            offset = relativedelta(months=int(t_composite_schema.temporal_composite_t))

            current_date = start_date
            current_date.replace(day=1)

            next_month_first = current_date + offset

            if end_date < current_date:
                print('Set end date to the end of month')
                end_date = current_date + offset

            while current_date < end_date:
                current_date_str = current_date.strftime('%Y-%m')

                for band, temporal in warp_merge.items():
                    requestedperiods.setdefault(band, dict())
                    requestedperiods[band].setdefault(current_date_str, dict())

                    for item_date in temporal:
                        scene_date = datetime.strptime(item_date, '%Y-%m-%d').date()
                        if scene_date >= current_date:
                            requestedperiods[band][current_date_str] = temporal

                current_date += offset

            return requestedperiods

        return

    @classmethod
    def search_stac(cls, collection_name: str, tiles: List[str], start_date: str, end_date: str):
        from stac import STAC

        stac_cli = STAC('http://brazildatacube.dpi.inpe.br/bdc-stac/0.7.0/')

        filter_opts = dict(
            time='{}/{}'.format(start_date, end_date),
            limit=100000
        )

        bbox_result = db.session.query(
            Tile.id,
            func.ST_AsText(func.ST_BoundingDiagonal(func.ST_Force2D(Tile.geom_wgs84)))
        ).filter(
            Tile.id.in_(tiles)
        ).all()

        result = dict()

        stac_collection = stac_cli.collection(collection_name)
        collection_bands = stac_collection['properties']['bdc:bands']

        for res in bbox_result:
            bbox_grs_schema = res[0]
            bbox = res[1][res[1].find('(') + 1:res[0].find(')')]
            bbox = bbox.replace(' ', ',')
            filter_opts['bbox'] = bbox
            items = stac_cli.collection_items(collection_name, filter=filter_opts)

            for band in collection_bands:
                for feature in items['features']:
                    if not feature['assets'].get(band):
                        continue

                    result.setdefault(band, dict())
                    feature_date_time_str = feature['properties']['datetime']
                    feature_date = datetime.strptime(feature_date_time_str, '%Y-%m-%dT%H:%M:%S').date()
                    feature_date_str = feature_date.strftime('%Y-%m-%d')

                    result[band].setdefault(feature_date_str, dict())

                    if feature['assets'].get(band):
                        collection_bands[band]['common_name'] = band
                        asset_definition = dict(
                            url=feature['assets'][band]['href'],
                            band=collection_bands[band],
                            scene_id=feature['id'],
                            tile=bbox_grs_schema,
                            feature_tile=feature['properties']['bdc:tile'],
                            datetime=feature['properties']['datetime']
                        )

                        result[band][feature_date_str][feature['id']] = asset_definition

        return result

    @staticmethod
    def create_activity(collection: str, scene: str, activity_type: str, scene_type: str, band: str, **parameters):
        return dict(
            band=band,
            collection_id=collection,
            activity_type=activity_type,
            tags=parameters.get('tags', []),
            sceneid=scene,
            scene_type=scene_type,
            args=parameters
        )

    @staticmethod
    def get_warped_datacube(datacube: str,
                            tiles: Optional[List[str]]=None,
                            start_date: Optional[str]=None,
                            end_date: Optional[str]=None):
        for fn in ['MEDIAN', 'STACK']:
            datacube = datacube.replace(fn, 'WARPED')

        try:
            return CubeBusiness.search_stac(datacube, tiles, start_date, end_date)
        except:
            return None

    @classmethod
    def maestro(cls, datacube, collections, tiles, start_date, end_date):
        from .maestro import Maestro

        maestro = Maestro(datacube, collections, tiles, start_date, end_date)

        maestro.orchestrate()

        maestro.dispatch_celery()

        return dict(ok=True)
