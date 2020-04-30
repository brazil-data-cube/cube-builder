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
from bdc_db.models import (Band, Collection, GrsSchema, RasterSizeSchema,
                           TemporalCompositionSchema, Tile)
from bdc_db.models.base_sql import BaseModel, db
from rasterio.warp import transform
from sqlalchemy import func
from werkzeug.exceptions import BadRequest, Conflict, NotFound

from .forms import (CollectionForm, GrsSchemaForm, RasterSchemaForm,
                    TemporalSchemaForm)
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
    def generate_periods(cls, schema, step, start_date=None, last_date=None, **kwargs):
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

    @classmethod
    def create_raster_schema(cls, grs_schema, resolution, chunk_size_x, chunk_size_y) -> Tuple[dict, int]:
        """Create Brazil Data Cube Raster Schema Size based in GrsSchema."""
        tile = db.session() \
            .query(
            Tile, func.ST_Xmin(Tile.geom), func.ST_Xmax(Tile.geom),
            func.ST_Ymin(Tile.geom), func.ST_Ymax(Tile.geom)
        ).filter(
            Tile.grs_schema_id == grs_schema
        ).first()
        if not tile:
            raise BadRequest('GrsSchema "{}" not found.'.format(grs_schema))

        # x = Xmax - Xmin || y = Ymax - Ymin
        raster_size_x = int(round((tile[2] - tile[1]) / int(resolution), 0))
        raster_size_y = int(round((tile[4] - tile[3]) / int(resolution), 0))

        raster_schema_id = '{}-{}'.format(grs_schema, resolution)

        raster_schema = RasterSizeSchema.query().filter(
            RasterSizeSchema.id == raster_schema_id
        ).first()

        if raster_schema is None:
            model = RasterSizeSchema(
                id=raster_schema_id,
                raster_size_x=raster_size_x,
                raster_size_y=raster_size_y,
                raster_size_t=1,
                chunk_size_x=chunk_size_x,
                chunk_size_y=chunk_size_y,
                chunk_size_t=1
            )

            model.save()

            return RasterSchemaForm().dump(model), 201

        raise Conflict('RasterSchema "{}" already exists.'.format(raster_schema_id))

    @classmethod
    def create_grs_schema(cls, name, description, projection, meridian, degreesx, degreesy, bbox):
        """Create a Brazil Data Cube Grid Schema."""
        bbox = bbox.split(',')
        bbox_obj = dict(
            w=float(bbox[0]),
            n=float(bbox[1]),
            e=float(bbox[2]),
            s=float(bbox[3])
        )
        tilesrsp4 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        if projection == 'aea':
            tilesrsp4 = "+proj=aea +lat_1=10 +lat_2=-40 +lat_0=0 +lon_0={} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(
                meridian)
        elif projection == 'sinu':
            tilesrsp4 = "+proj=sinu +lon_0={0} +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs".format(0.)

        # Number of tiles and base tile
        numtilesx = int(360. / degreesx)
        numtilesy = int(180. / degreesy)
        hBase = numtilesx / 2
        vBase = numtilesy / 2

        # Tile size in meters (dx,dy) at center of system (argsmeridian,0.)
        src_crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        dst_crs = tilesrsp4
        xs = [(meridian - degreesx / 2), (meridian + degreesx / 2), meridian, meridian, 0.]
        ys = [0., 0., -degreesy / 2, degreesy / 2, 0.]
        out = transform(src_crs, dst_crs, xs, ys, zs=None)
        x1 = out[0][0]
        x2 = out[0][1]
        y1 = out[1][2]
        y2 = out[1][3]
        dx = x2 - x1
        dy = y2 - y1

        # Coordinates of WRS center (0.,0.) - top left coordinate of (hBase,vBase)
        xCenter = out[0][4]
        yCenter = out[1][4]
        # Border coordinates of WRS grid
        xMin = xCenter - dx * hBase
        yMax = yCenter + dy * vBase

        # Upper Left is (xl,yu) Bottom Right is (xr,yb)
        xs = [bbox_obj['w'], bbox_obj['e'], meridian, meridian]
        ys = [0., 0., bbox_obj['n'], bbox_obj['s']]
        out = transform(src_crs, dst_crs, xs, ys, zs=None)
        xl = out[0][0]
        xr = out[0][1]
        yu = out[1][2]
        yb = out[1][3]
        hMin = int((xl - xMin) / dx)
        hMax = int((xr - xMin) / dx)
        vMin = int((yMax - yu) / dy)
        vMax = int((yMax - yb) / dy)

        # Insert grid
        model = GrsSchema.query().filter(
            GrsSchema.id == name
        ).first()

        if not model:
            model = GrsSchema(
                id=name,
                description=description,
                crs=tilesrsp4
            )
            model.save()

        tiles = []
        dst_crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        src_crs = tilesrsp4
        for ix in range(hMin, hMax + 1):
            x1 = xMin + ix * dx
            x2 = x1 + dx
            for iy in range(vMin, vMax + 1):
                y1 = yMax - iy * dy
                y2 = y1 - dy
                # Evaluate the bounding box of tile in longlat
                xs = [x1, x2, x2, x1]
                ys = [y1, y1, y2, y2]
                out = transform(src_crs, dst_crs, xs, ys, zs=None)
                UL_lon = out[0][0]
                UL_lat = out[1][0]
                UR_lon = out[0][1]
                UR_lat = out[1][1]
                LR_lon = out[0][2]
                LR_lat = out[1][2]
                LL_lon = out[0][3]
                LL_lat = out[1][3]

                wkt_wgs84 = 'POLYGON(({} {},{} {},{} {},{} {},{} {}))'.format(
                    UL_lon, UL_lat,
                    UR_lon, UR_lat,
                    LR_lon, LR_lat,
                    LL_lon, LL_lat,
                    UL_lon, UL_lat)

                wkt = 'POLYGON(({} {},{} {},{} {},{} {},{} {}))'.format(
                    x1, y2,
                    x2, y2,
                    x2, y1,
                    x1, y1,
                    x1, y2)

                # Insert tile
                tiles.append(Tile(
                    id='{0:03d}{1:03d}'.format(ix, iy),
                    grs_schema_id=name,
                    geom_wgs84='SRID=4326;{}'.format(wkt_wgs84),
                    geom='SRID=0;{}'.format(wkt),
                    min_x=x1,
                    max_y=y1
                ))

        BaseModel.save_all(tiles)

        return GrsSchemaForm().dump(model), 201
