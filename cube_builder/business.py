#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder business interface."""

from datetime import datetime
from typing import List, Tuple

# 3rdparty
from bdc_db.models import (Band, Collection, CollectionItem, CompositeFunctionSchema, GrsSchema,
                           RasterSizeSchema, TemporalCompositionSchema, Tile)
from bdc_db.models.base_sql import BaseModel, db
from geoalchemy2.shape import from_shape
from rasterio.warp import transform
from shapely.geometry import Polygon
from sqlalchemy import func, JSON, String
from werkzeug.exceptions import BadRequest, Conflict, NotFound

from .constants import (CLEAR_OBSERVATION_NAME, CLEAR_OBSERVATION_ATTRIBUTES,
                        PROVENANCE_NAME, PROVENANCE_ATTRIBUTES,
                        TOTAL_OBSERVATION_NAME, TOTAL_OBSERVATION_ATTRIBUTES)
from .forms import (CollectionForm, GrsSchemaForm, RasterSchemaForm,
                    TemporalSchemaForm, CompositeFunctionForm, BandForm, CollectionItemForm)
from .image import validate_merges
from .maestro import Maestro, decode_periods
from .models import Activity
from .utils import get_cube_parts, get_or_create_model, get_cube_id


class CubeBusiness:
    """Define Cube Builder interface for data cube creation."""

    @classmethod
    def get_or_create_band(cls, cube, name, common_name, min, max,
                           fill, data_type, res_x, res_y, scale, description=None) -> Band:
        """Get or try to create a data cube band on database.

        Notes:
            When band not found, it adds a new Band object to the SQLAlchemy db.session.
            You may need to commit the session after execution.

        Returns:
            A SQLAlchemy band object.
        """
        where = dict(
            collection_id=cube,
            name=name
        )

        defaults = dict(
            min=min,
            max=max,
            fill=fill,
            data_type=data_type,
            description=description,
            scale=scale,
            common_name=common_name,
            resolution_x=res_x,
            resolution_y=res_y,
            resolution_unit='m'
        )

        band, _ = get_or_create_model(Band, defaults=defaults, **where)

        return band

    @classmethod
    def _create_cube_definition(cls, cube_id: str, params: dict) -> dict:
        """Create a data cube definition.

        Basically, the definition consists in `Collection` and `Band` attributes.

        Note:
            It does not try to create when data cube already exists.

        Args:
            cube_id - Data cube
            params - Dict of required values to create data cube. See @validators.py

        Returns:
            A serialized data cube information.
        """
        cube_parts = get_cube_parts(cube_id)

        function = cube_parts.composite_function

        raster_size_id = '{}-{}'.format(params['grs'], int(params['resolution']))

        cube_id = cube_parts.datacube

        cube = Collection.query().filter(Collection.id == cube_id).first()

        if cube is None:
            cube = Collection(
                id=cube_id,
                temporal_composition_schema_id=params['temporal_schema'] if function.upper() != 'IDENTITY' else 'Anull',
                raster_size_schema_id=raster_size_id,
                composite_function_schema_id=function,
                grs_schema_id=params['grs'],
                description=params['description'],
                radiometric_processing=None,
                geometry_processing=None,
                sensor=None,
                is_cube=True,
                oauth_scope=params.get('oauth_scope', None),
                license=params['license'],
                bands_quicklook=','.join(params['bands_quicklook'])
            )

            cube.save(commit=False)

            bands = []

            default_bands = (CLEAR_OBSERVATION_NAME.lower(), TOTAL_OBSERVATION_NAME.lower(), PROVENANCE_NAME.lower())

            for band in params['bands']:
                name = band['name'].strip()

                if name in default_bands:
                    continue

                is_not_cloud = params['quality_band'] != band['name']

                if band['name'] == params['quality_band']:
                    data_type = 'uint8'
                else:
                    data_type = band['data_type']

                band_model = Band(
                    name=name,
                    collection_id=cube.id,
                    min=0,
                    max=10000 if is_not_cloud else 4,
                    fill=-9999 if is_not_cloud else 255,
                    scale=0.0001 if is_not_cloud else 1,
                    data_type=data_type,
                    common_name=band['common_name'],
                    resolution_x=params['resolution'],
                    resolution_y=params['resolution'],
                    resolution_unit='m',
                    description='',
                    mime_type='image/tiff'
                )
                band_model.save(commit=False)
                bands.append(band_model)

        # Create default Cube Bands
        if function != 'IDENTITY':
            _ = cls.get_or_create_band(cube_id, **CLEAR_OBSERVATION_ATTRIBUTES,
                                       res_x=params['resolution'], res_y=params['resolution'])
            _ = cls.get_or_create_band(cube_id, **TOTAL_OBSERVATION_ATTRIBUTES,
                                       res_x=params['resolution'], res_y=params['resolution'])

            if function == 'STK':
                _ = cls.get_or_create_band(cube_id, **PROVENANCE_ATTRIBUTES,
                                           res_x=params['resolution'], res_y=params['resolution'])

        return CollectionForm().dump(cube)

    @classmethod
    def create(cls, params):
        """Create a data cube definition.

        Note:
            If you provide a data cube with composite function like MED, STK, it creates
            the cube metadata IDENTITY and the given function name.

        Returns:
             Tuple with serialized cube and HTTP Status code, respectively.
        """
        cube_name = '{}_{}'.format(
            params['datacube'],
            int(params['resolution'])
        )

        params['bands'].extend(params['indexes'])

        with db.session.begin_nested():
            # Create data cube IDENTITY
            cube = cls._create_cube_definition(cube_name, params)

            cube_serialized = [cube]

            if params['composite_function'] != 'IDENTITY':
                temporal_schema = TemporalCompositionSchema.query() \
                    .filter(TemporalCompositionSchema.id == params['temporal_schema']) \
                    .first_or_404()

                temporal_str = f'{temporal_schema.temporal_composite_t}{temporal_schema.temporal_composite_unit[0].upper()}'

                cube_name_composite = f'{cube_name}_{temporal_str}_{params["composite_function"]}'

                # Create data cube with temporal composition
                cube_composite = cls._create_cube_definition(cube_name_composite, params)
                cube_serialized.append(cube_composite)

        db.session.commit()

        return cube_serialized, 201

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
    def check_for_invalid_merges(cls, data_cube: str, tile_id: str, start: str, end: str) -> Tuple[dict, int]:
        """List merge files used in data cube and check for invalid scenes.

        Args:
            data_cube: Data cube name
            tile_id: Brazil Data Cube Tile identifier
            start: Activity start date (period)
            end: Activity End (period)

        Returns:
            List of Images used in period
        """
        cube = Collection.query().filter(Collection.id == data_cube).first()

        if cube is None or not cube.is_cube:
            raise NotFound('Cube {} not found'.format(data_cube))

        # TODO validate schema to avoid start/end too abroad

        res = Activity.list_merge_files(cube, tile_id, start, end)

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
    def generate_periods(cls, schema, step, start=None, end=None, **kwargs) -> List[str]:
        """Generate data cube periods using temporal composition schema.

        Args:
            schema: Temporal Schema (M, A)
            step: Temporal Step
            start: Start date offset. Default is '2016-01-01'.
            end: End data offset. Default is '2019-12-31'
            **kwargs: Optional parameters

        Returns:
            List of periods between start/last date
        """
        start_date = start or '2016-01-01'
        last_date = end or '2019-12-31'

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
        bbox_obj = {
            "w": float(bbox[0]),
            "n": float(bbox[1]),
            "e": float(bbox[2]),
            "s": float(bbox[3])
        }
        tile_srs_p4 = "+proj=longlat +ellps=GRS80 +datum=GRS80 +no_defs"
        if projection == 'aea':
            tile_srs_p4 = "+proj=aea +lat_1=-1 +lat_2=-29 +lat_0=0 +lon_0={} +x_0=0 +y_0=0 +ellps=GRS80 +datum=GRS80 +units=m +no_defs".format(
                meridian)
        elif projection == 'sinu':
            tile_srs_p4 = "+proj=sinu +lon_0={} +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs".format(
                meridian)

        # Number of tiles and base tile
        num_tiles_x = int(360. / degreesx)
        num_tiles_y = int(180. / degreesy)
        h_base = num_tiles_x / 2
        v_base = num_tiles_y / 2

        # Tile size in meters (dx,dy) at center of system (argsmeridian,0.)
        src_crs = '+proj=longlat +ellps=GRS80 +datum=GRS80 +no_defs'
        dst_crs = tile_srs_p4
        xs = [(meridian - degreesx / 2), (meridian + degreesx / 2), meridian, meridian, 0.]
        ys = [0., 0., -degreesy / 2, degreesy / 2, 0.]
        out = transform(src_crs, dst_crs, xs, ys, zs=None)
        x1 = out[0][0]
        x2 = out[0][1]
        y1 = out[1][2]
        y2 = out[1][3]
        dx = x2 - x1
        dy = y2 - y1

        # Coordinates of WRS center (0.,0.) - top left coordinate of (h_base,v_base)
        x_center = out[0][4]
        y_center = out[1][4]
        # Border coordinates of WRS grid
        x_min = x_center - dx * h_base
        y_max = y_center + dy * v_base

        # Upper Left is (xl,yu) Bottom Right is (xr,yb)
        xs = [bbox_obj['w'], bbox_obj['e'], meridian, meridian]
        ys = [0., 0., bbox_obj['n'], bbox_obj['s']]
        out = transform(src_crs, dst_crs, xs, ys, zs=None)
        xl = out[0][0]
        xr = out[0][1]
        yu = out[1][2]
        yb = out[1][3]
        h_min = int((xl - x_min) / dx)
        h_max = int((xr - x_min) / dx)
        v_min = int((y_max - yu) / dy)
        v_max = int((y_max - yb) / dy)

        # Insert grid
        grs = GrsSchema.query().filter(
            GrsSchema.id == name
        ).first()
        if not grs:
            grs = GrsSchema(
                id=name,
                description=description,
                crs=tile_srs_p4
            ).save()

        tiles = []
        dst_crs = '+proj=longlat +ellps=GRS80 +datum=GRS80 +no_defs'
        src_crs = tile_srs_p4
        for ix in range(h_min, h_max + 1):
            x1 = x_min + ix * dx
            x2 = x1 + dx
            for iy in range(v_min, v_max + 1):
                y1 = y_max - iy * dy
                y2 = y1 - dy
                # Evaluate the bounding box of tile in longlat
                xs = [x1, x2, x2, x1]
                ys = [y1, y1, y2, y2]
                out = transform(src_crs, dst_crs, xs, ys, zs=None)
                ul_lon = out[0][0]
                ul_lat = out[1][0]
                ur_lon = out[0][1]
                ur_lat = out[1][1]
                lr_lon = out[0][2]
                lr_lat = out[1][2]
                ll_lon = out[0][3]
                ll_lat = out[1][3]

                poly_wgs84 = from_shape(
                    Polygon(
                        [
                            (ul_lon, ul_lat),
                            (ur_lon, ur_lat),
                            (lr_lon, lr_lat),
                            (ll_lon, ll_lat),
                            (ul_lon, ul_lat)
                        ]
                    ),
                    srid=4674
                )

                poly_aea = from_shape(
                    Polygon(
                        [
                            (x1, y2),
                            (x2, y2),
                            (x2, y1),
                            (x1, y1),
                            (x1, y2)
                        ]
                    ),
                    srid=0
                )

                # Insert tile
                tiles.append(Tile(
                    id='{0:03d}{1:03d}'.format(ix, iy),
                    grs_schema_id=name,
                    geom_wgs84=poly_wgs84,
                    geom=poly_aea
                ))

        BaseModel.save_all(tiles)

        data = GrsSchemaForm().dump(grs)
        data['tiles'] = [t.id for t in tiles]

        return data, 201

    @classmethod
    def list_raster_size(cls):
        results = RasterSizeSchema.query().all()

        return RasterSchemaForm().dump(results, many=True)

    @classmethod
    def get_cube_status(cls, datacube: str) -> Tuple[dict, int]:
        _ = cls.get_cube(datacube)
        # TODO: Retrieve cube status based in Activity. We must check if the activity table is storing start/end dates
        return dict(
            finished=True,
            start_date=str(datetime.utcnow()),
            last_date=str(datetime.utcnow()),
            done=0,
            duration='',
            collection_item=1
        ), 200

    @classmethod
    def get_cube_or_404(cls, datacube: str) -> Collection:
        return Collection.query().filter(Collection.is_cube.is_(True), Collection.id == datacube).first_or_404()

    @classmethod
    def get_cube(cls, datacube: str) -> dict:
        cube = cls.get_cube_or_404(datacube)

        temporal = db.session.query(
            func.min(CollectionItem.composite_start).cast(String),
            func.max(CollectionItem.composite_end).cast(String)
        ).filter(CollectionItem.collection_id == cube.id).first()

        temporal_composition = dict(
            schema=cube.temporal_composition_schema.temporal_schema,
            unit=cube.temporal_composition_schema.temporal_composite_unit,
            step=cube.temporal_composition_schema.temporal_composite_t
        )

        bands = Band.query().filter(Band.collection_id == cube.id).all()

        if temporal is None:
            temporal = []

        dump_collection = CollectionForm().dump(cube)

        dump_collection['temporal'] = temporal
        dump_collection['bands'] = BandForm().dump(bands, many=True)
        dump_collection['temporal_composition'] = temporal_composition

        return dump_collection

    @classmethod
    def list_cubes(cls) -> List[dict]:
        cubes = Collection.query().filter(Collection.is_cube.is_(True)).all()

        serializer = CollectionForm()

        output = []

        for cube in cubes:
            cube_dict = serializer.dump(cube)
            cube_dict['status'] = 'pending'

            output.append(cube_dict)

        return output

    @classmethod
    def list_tiles_cube(cls, datacube: str) -> List[dict]:
        cube = cls.get_cube_or_404(datacube)

        features = db.session.query(
            func.ST_AsGeoJSON(func.ST_SetSRID(Tile.geom_wgs84, 4326), 6, 3).cast(JSON)
        ).distinct(Tile.id).filter(
            CollectionItem.tile_id == Tile.id,
            CollectionItem.collection_id == cube.id,
            Tile.grs_schema_id == cube.grs_schema_id
        ).all()

        return [feature[0] for feature in features]

    @classmethod
    def get_grs_schema(cls, grs_id: str) -> dict:
        grs = GrsSchema.query().get_or_404(grs_id)

        return GrsSchemaForm().dump(grs)

    @classmethod
    def list_grs_schemas(cls) -> dict:
        grs_list = GrsSchema.query().all()

        return GrsSchemaForm().dump(grs_list, many=True)

    @classmethod
    def list_temporal_composition(cls) -> List[dict]:
        temporal_compositions = TemporalCompositionSchema.query().all()

        return TemporalSchemaForm().dump(temporal_compositions, many=True)

    @classmethod
    def list_composite_functions(cls) -> List[dict]:
        compositions = CompositeFunctionSchema.query().all()

        return CompositeFunctionForm().dump(compositions, many=True)

    @classmethod
    def list_cube_items_tiles(cls, cube: str) -> List[str]:
        """Retrieve the tiles from collection items built."""
        tiles = db.session.query(CollectionItem.tile_id)\
            .filter(CollectionItem.collection_id == cube)\
            .group_by(CollectionItem.tile_id)\
            .all()

        return [t.tile_id for t in tiles]

    @classmethod
    def get_cube_meta(cls, cube_name: str) -> dict:
        """Retrieve the data cube metadata used to build a data cube items.

        The metadata includes:
        - STAC provider url
        - Collection used to generate.

        Note:
            When there is no data cube item generated yet, raises BadRequest.
        """
        cube = CubeBusiness.get_cube_or_404(cube_name)

        identity_cube = get_cube_id(cube.id)

        merge_activity: Activity = Activity.query()\
            .filter(Activity.activity_type == 'MERGE', Activity.warped_collection_id == identity_cube)\
            .first()

        if merge_activity is None:
            raise BadRequest('There is no data cube activity')

        return dict(
            url_stac='',
            collections=merge_activity.args['dataset'],
            satellite='',
        )

    @classmethod
    def list_cube_items(cls, data_cube: str, bbox: str = None, start: str = None,
                        end: str = None, tiles: str = None, page: int = 1, per_page: int = 10):
        cube = CubeBusiness.get_cube_or_404(data_cube)

        where = [
            CollectionItem.collection_id == data_cube,
            Tile.grs_schema_id == cube.grs_schema_id,
            Tile.id == CollectionItem.tile_id
        ]

        if start:
            where.append(CollectionItem.composite_start >= start)

        if end:
            where.append(CollectionItem.composite_end <= end)

        if tiles:
            tiles = tiles.split(',') if isinstance(tiles, str) else tiles
            where.append(
                Tile.id.in_(tiles)
            )

        if bbox:
            xmin, ymin, xmax, ymax = [float(coord) for coord in bbox.split(',')]
            where.append(
                func.ST_Intersects(
                    func.ST_SetSRID(Tile.geom_wgs84, 4326), func.ST_MakeEnvelope(xmin, ymin, xmax, ymax, 4326)
                )
            )

        paginator = db.session.query(CollectionItem).filter(
            *where
        ).order_by(CollectionItem.item_date.desc()).paginate(int(page), int(per_page), error_out=False)

        result = []

        serializer = CollectionItemForm()

        for item in paginator.items:
            obj = serializer.dump(item)
            obj['quicklook'] = item.quicklook

            result.append(obj)

        return dict(
            items=result,
            page=page,
            per_page=page,
            total_items=paginator.total,
            total_pages=paginator.pages
        )
