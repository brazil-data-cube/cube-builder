#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#


"""Define Cube Builder business interface."""
from copy import deepcopy
from datetime import datetime
from typing import Tuple, Union

# 3rdparty
import sqlalchemy
from bdc_catalog.models import (Band, BandSRC, Collection, CollectionSRC, CompositeFunction, GridRefSys, Item, MimeType,
                                Quicklook, ResolutionUnit, SpatialRefSys, Tile, db)
from geoalchemy2 import func
from rasterio.crs import CRS
from werkzeug.exceptions import NotFound, abort

from .constants import (CLEAR_OBSERVATION_ATTRIBUTES, CLEAR_OBSERVATION_NAME, COG_MIME_TYPE, DATASOURCE_ATTRIBUTES,
                        IDENTITY, PROVENANCE_ATTRIBUTES, PROVENANCE_NAME, TOTAL_OBSERVATION_ATTRIBUTES,
                        TOTAL_OBSERVATION_NAME)
from .forms import CollectionForm
from .grids import create_grids
from .models import Activity, CubeParameters
from .utils import get_srid_column
from .utils.image import validate_merges
from .utils.processing import get_or_create_model
from .utils.serializer import Serializer
from .utils.timeline import Timeline


class CubeController:
    """Define Cube Builder interface for data cube creation."""

    @staticmethod
    def get_cube_or_404(cube_id: Union[int, str] = None):
        """Try to retrieve a data cube on database and raise 404 when not found."""
        return Collection.get_by_id(cube_id)

    @classmethod
    def get_or_create_band(cls, cube, name, common_name, min_value, max_value,
                           nodata, data_type, resolution_x, resolution_y, scale,
                           resolution_unit_id, mime_type_id, description=None, scale_add=0) -> Band:
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
            min_value=min_value,
            max_value=max_value,
            nodata=nodata,
            data_type=data_type,
            description=description,
            scale_mult=scale,
            scale_add=scale_add,
            mime_type_id=mime_type_id,
            common_name=common_name,
            resolution_unit_id=resolution_unit_id
        )

        band, created = get_or_create_model(Band, defaults=defaults, **where)
        if created:
            band.add_eo_meta(resolution_x=resolution_x, resolution_y=resolution_y)

        return band

    @staticmethod
    def _validate_band_metadata(metadata: dict, band_map: dict) -> dict:
        bands = []

        for band in metadata['expression']['bands']:
            bands.append(band_map[band].id)

        metadata['expression']['bands'] = bands

        return metadata

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
        function = params['composite_function']

        cube = (
            Collection.query()
            .filter(Collection.identifier == f'{cube_id}-{params["version"]}')
            .first()
        )

        grs = GridRefSys.query().filter(GridRefSys.name == params['grs']).first()

        if grs is None:
            abort(404, f'Grid {params["grs"]} not found.')

        cube_function = CompositeFunction.query().filter(CompositeFunction.alias == function).first()

        if cube_function is None:
            abort(404, f'Function {function} not found.')

        data = dict(name='Meter', symbol='m')
        resolution_meter, _ = get_or_create_model(ResolutionUnit, defaults=data, symbol='m')

        mime_type, _ = get_or_create_model(MimeType, defaults=dict(name=COG_MIME_TYPE), name=COG_MIME_TYPE)

        if cube is None:
            cube = Collection(
                name=cube_id,
                title=params['title'],
                temporal_composition_schema=params['temporal_composition'] if function != IDENTITY else None,
                composite_function_id=cube_function.id,
                grs=grs,
                metadata_=params['metadata'],
                category=params.get('category') or 'eo',
                description=params['description'],
                collection_type='cube',
                is_available=params.get('public', False),
                version=params['version'],
                keywords=params.get('keywords'),
                summaries=params.get('summaries'),
            )

            cube.save(commit=False)

            bands = []

            default_bands = (CLEAR_OBSERVATION_NAME.lower(), TOTAL_OBSERVATION_NAME.lower(), PROVENANCE_NAME.lower())

            band_map = dict()

            for band in params['bands']:
                name = band['name'].strip()

                if name.lower() in default_bands:
                    continue

                is_not_cloud = params['quality_band'] != band['name'] if params.get('quality_band') is not None else False

                data_type = band['data_type']

                band_model = Band(
                    name=name,
                    common_name=band['common_name'],
                    collection=cube,
                    min_value=-10000 if band.get('metadata') else 0,
                    max_value=10000,
                    nodata=band['nodata'],
                    scale_mult=0.0001 if is_not_cloud else 1,
                    scale_add=params.get('scale_add'),
                    data_type=data_type,
                    resolution_unit_id=resolution_meter.id,
                    description='',
                    mime_type_id=mime_type.id
                )

                band_model.add_eo_meta(resolution_x=params['resolution'], resolution_y=params['resolution'])

                if band.get('metadata'):
                    band_model.metadata_.update(cls._validate_band_metadata(deepcopy(band['metadata']), band_map))

                band_model.save(commit=False)
                bands.append(band_model)

                band_map[name] = band_model

                if band_model.metadata_ and band_model.metadata_.get('expression'):
                    for _band_origin_id in band_model.metadata_['expression']['bands']:
                        band_provenance = BandSRC(band_src_id=_band_origin_id, band_id=band_model.id)
                        band_provenance.save(commit=False)

            quicklook = Quicklook(red=band_map[params['bands_quicklook'][0]].id,
                                  green=band_map[params['bands_quicklook'][1]].id,
                                  blue=band_map[params['bands_quicklook'][2]].id,
                                  collection=cube)

            quicklook.save(commit=False)

        default_params = dict(
            metadata_=params['parameters']
        )

        cube.item_assets = _make_item_assets(cube)

        default_params['metadata_']['quality_band'] = params['quality_band']
        cube_parameters, _ = get_or_create_model(CubeParameters, defaults=default_params, collection_id=cube.id)
        db.session.add(cube_parameters)

        # Create default Cube Bands
        if function != IDENTITY:
            _ = cls.get_or_create_band(cube.id, **CLEAR_OBSERVATION_ATTRIBUTES, mime_type_id=mime_type.id,
                                       resolution_unit_id=resolution_meter.id,
                                       resolution_x=params['resolution'], resolution_y=params['resolution'])
            _ = cls.get_or_create_band(cube.id, **TOTAL_OBSERVATION_ATTRIBUTES, mime_type_id=mime_type.id,
                                       resolution_unit_id=resolution_meter.id,
                                       resolution_x=params['resolution'], resolution_y=params['resolution'])

            if function in ('STK', 'LCF'):
                _ = cls.get_or_create_band(cube.id, **PROVENANCE_ATTRIBUTES, mime_type_id=mime_type.id,
                                           resolution_unit_id=resolution_meter.id,
                                           resolution_x=params['resolution'], resolution_y=params['resolution'])

        if params.get('is_combined') and function != 'MED':
            _ = cls.get_or_create_band(cube.id, **DATASOURCE_ATTRIBUTES, mime_type_id=mime_type.id,
                                       resolution_unit_id=resolution_meter.id,
                                       resolution_x=params['resolution'], resolution_y=params['resolution'])

        return CollectionForm().dump(cube)

    @classmethod
    def create(cls, params):
        """Create a data cube definition.

        Note:
            If you provide a data cube with composite function like MED, STK, it creates
            the cube metadata Identity and the given function name.

        Returns:
             Tuple with serialized cube and HTTP Status code, respectively.
        """
        cube_name = cube_identity = params['datacube']
        # When custom idt name given, use it.
        if params.get('datacube_identity'):
            cube_identity = params['datacube_identity']

        params['bands'].extend(params['indexes'])

        with db.session.begin_nested():
            # Create data cube Identity
            identity_params = deepcopy(params)
            identity_params['composite_function'] = IDENTITY
            cube = cls._create_cube_definition(cube_identity, identity_params)

            cube_serialized = [cube]

            if params['composite_function'] != IDENTITY:
                if cube_name == cube_identity:
                    abort(409, f'Duplicated data cube name for cube: '
                               f'Composed {cube_name} and Identity {cube_identity}')

                # Create data cube with temporal composition
                cube_composite = cls._create_cube_definition(cube_name, params)
                cube_serialized.append(cube_composite)

                # Create relationship between identity and composed
                collection_src_opts = {
                    'collection_id': cube_composite['id'],
                    'collection_src_id': cube['id']
                }
                _, _ = get_or_create_model(CollectionSRC, defaults=collection_src_opts, **collection_src_opts)

        db.session.commit()

        return cube_serialized, 201

    @classmethod
    def update(cls, cube_id: int, params):
        """Update data cube definition.

        Returns:
             Tuple with serialized cube and HTTP Status code, respectively.
        """
        with db.session.begin_nested():
            cube = cls.get_cube_or_404(cube_id=cube_id)

            cube.title = params.get('title') or cube.title
            cube.metadata_ = params.get('metadata') or cube.metadata_
            cube.description = params.get('description') or cube.description
            cube.is_available = params.get('is_available') or cube.is_available

            if params.get('bands'):
                band_map = {b.id: b for b in cube.bands}
                for band_meta in params['bands']:
                    if band_meta.get('id') not in band_map or band_meta.get('collection_id') != cube.id:
                        abort(400, f'Band "{band_meta.get("id")}" does not belongs to cube "{cube.name}-{cube.version}"')

                    band_ctx = band_map[band_meta['id']]
                    for prop, value in band_meta.items():
                        setattr(band_ctx, prop, value)

        db.session.commit()

        return {'message': 'Updated cube!'}, 200

    @classmethod
    def get_cube(cls, cube_id: int):
        """Retrieve a data cube definition metadata."""
        cube = cls.get_cube_or_404(cube_id=int(cube_id))

        dump_cube = Serializer.serialize(cube)
        dump_cube['bands'] = [Serializer.serialize(b) for b in cube.bands]
        dump_cube['quicklook'] = [
            list(filter(lambda b: b.id == cube.quicklook[0].red, cube.bands))[0].name,
            list(filter(lambda b: b.id == cube.quicklook[0].green, cube.bands))[0].name,
            list(filter(lambda b: b.id == cube.quicklook[0].blue, cube.bands))[0].name
        ]
        dump_cube['spatial_extent'] = None
        dump_cube['grid'] = cube.grs.name
        dump_cube['composite_function'] = dict(
            name=cube.composite_function.name,
            description=cube.composite_function.description,
            alias=cube.composite_function.alias,
        )
        dump_cube['summary'] = cls.summarize(cube)

        # Retrieve Item Start/End Date
        # This step is required since the default generation trigger
        # does not compute stats when is_available = False
        stats = (
            db.session
            .query(func.min(Item.start_date).label('start_date'),
                   func.max(Item.end_date).label('end_date'))
            .filter(Item.collection_id == cube.id)
            .first()
        )
        if stats:
            dump_cube['start_date'] = stats.start_date
            dump_cube['end_date'] = stats.end_date

        return dump_cube, 200

    @classmethod
    def list_cubes(cls, name: str = None, collection_type: str = None, public: bool = True):
        """Retrieve the list of data cubes from Brazil Data Cube database."""
        where = [Collection.collection_type.in_(['cube', 'mosaic'])]
        if collection_type and collection_type != 'all':
            where = [Collection.collection_type == collection_type]

        where.append(Collection.is_public.is_(public))

        if name:
            where.append(Collection.identifier.like(f'%{name}%'))

        cubes = Collection.query().filter(*where).order_by(Collection.id).all()

        serializer = CollectionForm()

        list_cubes = []

        for cube in cubes:
            cube_dict = serializer.dump(cube)

            # TODO: count activities from database and compare with execution
            count_tasks = 0

            cube_dict['status'] = 'Finished' if count_tasks == 0 else 'Pending'

            list_cubes.append(cube_dict)

        return list_cubes, 200

    @classmethod
    def get_cube_status(cls, cube_name: str):
        """Retrieve a data cube status, which includes total items, tiles, etc."""
        cube = cls.get_cube_or_404(cube_name)

        dates = (
            db.session.query(
                sqlalchemy.func.min(Activity.created), sqlalchemy.func.max(Activity.created)
            )
            .filter(Activity.collection_id == cube_name)
            .first()
        )

        count_items = db.session.query(Item.id).filter(Item.collection_id == cube.id).count()

        count_tasks = 0

        if count_tasks > 0:
            return dict(
                finished=False,
                done=0,
                not_done=count_tasks,
                error=0
            )

        return dict(
            finished=True,
            start_date=str(dates[0]),
            last_date=str(dates[1]),
            done=0,
            error=0,
            collection_item=count_items
        )

    @classmethod
    def list_tiles_cube(cls, cube_id: int, only_ids=False):
        """Retrieve all tiles (as GeoJSON) that belongs to a data cube."""
        features = db.session.query(
            Item.tile_id,
            Tile,
            func.ST_AsGeoJSON(Item.bbox, 6, 3).cast(sqlalchemy.JSON).label('geom')
        ).distinct(Item.tile_id).filter(Item.collection_id == cube_id, Item.tile_id == Tile.id).all()

        return [feature.Tile.name if only_ids else feature.geom for feature in features], 200

    @classmethod
    def maestro(cls, datacube, collections, tiles, start_date, end_date, **properties):
        """Search and Dispatch datacube generation on cluster.

        Args:
            datacube - Data cube identifier.
            collections - List of collections used to generate datacube.
            tiles - List of tiles to generate.
            start_date - Start period
            end_date - End period
            **properties - Additional properties used on datacube generation, such bands and cache.
        """
        from .maestro import Maestro

        maestro = Maestro(datacube, collections, tiles, start_date, end_date, **properties)

        maestro.orchestrate()

        maestro.run()

        return dict(ok=True)

    @classmethod
    def check_for_invalid_merges(cls, cube_id: str, tile_id: str, start_date: str,
                                 end_date: str) -> Tuple[dict, int]:
        """List merge files used in data cube and check for invalid scenes.

        Args:
            cube_id: Data cube name
            tile_id: Brazil Data Cube Tile identifier
            start_date: Activity start date (period)
            end_date: Activity End (period)

        Returns:
            List of Images used in period
        """
        cube = cls.get_cube_or_404(cube_id=cube_id)

        if not cube:
            raise NotFound('Cube {} not found'.format(cube_id))

        # TODO validate schema to avoid start/end too abroad
        identity = cube.composite_function.alias == IDENTITY

        res = Activity.list_merge_files(cube.name, tile_id, start_date[:10], end_date[:10], identity)

        result = validate_merges(res)

        return result, 200

    @classmethod
    def generate_periods(cls, schema, step, unit, start_date=None, last_date=None, cycle=None, intervals=None):
        """Generate data cube periods using temporal composition schema.

        Args:
            schema: Temporal Schema (continuous, cyclic)
            step: Temporal Step
            unit: Temporal Unit (day, month, year)
            start_date: Start date offset. Default is '2016-01-01'.
            last_date: End data offset. Default is '2019-12-31'

        Returns:
            List of periods between start/last date
        """
        start_date = datetime.strptime((start_date or '2016-01-01')[:10], '%Y-%m-%d').date()
        last_date = datetime.strptime((last_date or '2019-12-31')[:10], '%Y-%m-%d').date()

        periods = Timeline(schema, start_date, last_date, unit, int(step), cycle, intervals).mount()

        return dict(
            timeline=[[str(period[0]), str(period[1])] for period in periods]
        )

    @classmethod
    def cube_meta(cls, cube_id: int):
        """Retrieve the data sets (collections) used to build the given data cube."""
        cube = cls.get_cube_or_404(cube_id=cube_id)

        activity = Activity.query().filter(Activity.collection_id == cube.name).first()

        cube_params = CubeParameters\
            .query()\
            .filter(CubeParameters.collection_id == cube.id)\
            .first_or_404(f'No data cube parameters for {cube.name}-{cube.version}')

        response = dict(**cube_params.metadata_)

        if activity is not None:
            response['collections'] = list(activity.args['datasets'])

        return response, 200

    @classmethod
    def list_grs_schemas(cls):
        """Retrieve a list of available Grid Schema on Brazil Data Cube database."""
        schemas = GridRefSys.query().all()

        return [dict(**Serializer.serialize(schema), crs=schema.crs) for schema in schemas], 200

    @classmethod
    def get_grs_schema(cls, grs_id, bbox: Tuple[float, float, float, float] = None, tiles=None):
        """Retrieve a Grid Schema definition with tiles associated."""
        schema: GridRefSys = GridRefSys.query().filter(GridRefSys.id == grs_id).first()

        if schema is None:
            return 'GRS {} not found.'.format(grs_id), 404

        geom_table = schema.geom_table
        srid_column = get_srid_column(geom_table.c, default_srid=4326)
        where = []
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            where.append(
                func.ST_Intersects(
                    func.ST_MakeEnvelope(x_min, y_min, x_max, y_max, 4326),
                    func.ST_Transform(func.ST_SetSRID(geom_table.c.geom, srid_column), 4326)
                )
            )

        if tiles:
            where.append(geom_table.c.tile.in_(tiles))

        tiles = db.session.query(
            geom_table.c.tile,
            func.ST_AsGeoJSON(
                func.ST_Transform(
                    func.ST_SetSRID(geom_table.c.geom, srid_column),
                    4326
                ), 6, 3
            ).cast(sqlalchemy.JSON).label('geom_wgs84')
        ).filter(*where).all()

        dump_grs = Serializer.serialize(schema)
        dump_grs['tiles'] = [dict(id=t.tile, geom_wgs84=t.geom_wgs84) for t in tiles]

        return dump_grs, 200

    @classmethod
    def create_grs_schema(cls, names, description, projection, meridian, shape, tile_factor, bbox, srid=100001):
        """Create a Brazil Data Cube Grid Schema."""
        grid_mapping, proj4 = create_grids(names=names, projection=projection, meridian=meridian,
                                           shape=shape,
                                           bbox=bbox, srid=srid,
                                           tile_factor=tile_factor)

        for name in names:
            grid = grid_mapping[name]

            with db.session.begin_nested():
                crs = CRS.from_proj4(proj4)
                data = dict(
                    auth_name='Albers Equal Area',
                    auth_srid=srid,
                    srid=srid,
                    srtext=crs.to_wkt(),
                    proj4text=proj4
                )

                spatial_index, _ = get_or_create_model(SpatialRefSys, defaults=data, srid=srid)

                try:
                    grs = GridRefSys.create_geometry_table(table_name=name,
                                                           features=grid['features'],
                                                           srid=srid)
                except RuntimeError:
                    abort(409, f'GRS / Table {name} already exists.')

                grs.description = description
                db.session.add(grs)
                for tile_obj in grid['tiles']:
                    tile = Tile(**tile_obj, grs=grs)
                    db.session.add(tile)
            db.session.commit()

        return 'Grids {} created with successfully'.format(names), 201

    @classmethod
    def list_cube_items(cls, cube_id: str, bbox: str = None, start: str = None,
                        end: str = None, tiles: str = None, page: int = 1, per_page: int = 10):
        """Retrieve all data cube items done."""
        _ = cls.get_cube_or_404(cube_id=cube_id)

        where = [
            Item.collection_id == cube_id,
            Tile.id == Item.tile_id
        ]

        # temporal filter
        if start:
            where.append(Item.start_date >= start)
        if end:
            where.append(Item.end_date <= end)

        # tile(string) filter
        if tiles:
            tiles = tiles.split(',') if isinstance(tiles, str) else tiles
            where.append(
                Tile.name.in_(tiles)
            )

        # spatial filter
        if bbox:
            xmin, ymin, xmax, ymax = [float(coord) for coord in bbox.split(',')]
            where.append(
                func.ST_Intersects(
                    func.ST_SetSRID(Item.bbox, 4326), func.ST_MakeEnvelope(xmin, ymin, xmax, ymax, 4326)
                )
            )

        paginator = db.session.query(Item).filter(
            *where
        ).order_by(Item.start_date.desc()).paginate(page=int(page), per_page=int(per_page), error_out=False)

        result = []
        for item in paginator.items:
            obj = Serializer.serialize(item)
            obj['bbox'] = None
            obj['footprint'] = None
            obj['tile_id'] = item.tile.name
            if item.assets.get('thumbnail'):
                obj['quicklook'] = item.assets['thumbnail']['href']
            del obj['assets']

            result.append(obj)

        return dict(
            items=result,
            page=page,
            per_page=page,
            total_items=paginator.total,
            total_pages=paginator.pages
        ), 200

    @classmethod
    def list_composite_functions(cls):
        """Retrieve a list of available Composite Functions on Brazil Data Cube database."""
        schemas = CompositeFunction.query().all()

        return [Serializer.serialize(schema) for schema in schemas], 200

    @classmethod
    def configure_parameters(cls, collection_id, **kwargs) -> dict:
        """Configure data cube parameters to be passed during the execution.

        Args:
            collection_id (int): Data Cube identifier
            **kwargs (dict): Map of values to be set as parameter.

        Returns:
            dict The serialized cube parameters instance object.
        """
        cube = CubeController.get_cube_or_404(cube_id=collection_id)

        defaults = dict(
            metadata_=kwargs
        )
        cube_parameters, _ = get_or_create_model(CubeParameters, defaults=defaults, collection_id=cube.id)

        with db.session.begin_nested():
            # We must create a new copy to make effect in SQLAlchemy
            meta = deepcopy(cube_parameters.metadata_)
            meta.update(**kwargs)
            cube_parameters.metadata_ = meta
            db.session.add(cube_parameters)

        db.session.commit()

        return Serializer.serialize(cube_parameters)

    @classmethod
    def summarize(cls, cube: Union[str, Collection]) -> dict:
        """Retrieve data cube summarization.

        This method consists in compute the tile statistics like total items per tile, etc.
        """
        if isinstance(cube, str):
            cube = CubeController.get_cube_or_404(cube_id=cube)

        summary_rows = (
            db.session.query(
                Tile.name.label('tile'), func.count(Item.name).label('total_items')
            )
            .join(Tile, Tile.id == Item.tile_id)
            .filter(Item.collection_id == cube.id)
            .group_by(Tile.name)
            .order_by(Tile.name)
            .all()
        )

        return {
            row.tile: row.total_items
            for row in summary_rows
        }


def _make_item_assets(cube: Collection) -> dict:
    definition = {}
    for band in cube.bands:
        definition[band.name] = dict(
            type=band.mime_type.name,
            title=band.description or f'Band {band.name}',
            roles=['data'],
        )

    if cube.quicklook:
        definition['thumbnail'] = dict(
            type='image/png',
            title='Thumbnail',
            roles=['thumbnail']
        )

    return definition
