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

"""Define Cube Builder forms used to validate both data input and data serialization."""

# Python
import datetime
import json
import logging
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import time
from typing import List

# 3rdparty
import numpy
import shapely.geometry
import sqlalchemy
from bdc_catalog.models import Band, Collection, CollectionSRC, GridRefSys, Tile, db
from celery import chain, group
from geoalchemy2 import func
from geoalchemy2.shape import to_shape
from stac import STAC

# Cube Builder
from werkzeug.exceptions import abort

from .celery.tasks import prepare_blend, warp_merge
from .config import Config
from .constants import CLEAR_OBSERVATION_NAME, DATASOURCE_NAME, IDENTITY, PROVENANCE_NAME, TOTAL_OBSERVATION_NAME
from .local_accessor import load_format
from .models import CubeParameters
from .utils import get_srid_column
from .utils.processing import get_or_create_activity
from .utils.timeline import Timeline


def days_in_month(date):
    """Retrieve days in month from date."""
    year = int(date.split('-')[0])
    month = int(date.split('-')[1])
    nday = int(date.split('-')[2])
    if month == 12:
        nmonth = 1
        nyear = year + 1
    else:
        nmonth = month + 1
        nyear = year
    ndate = '{0:4d}-{1:02d}-{2:02d}'.format(nyear,nmonth,nday)
    td = numpy.datetime64(ndate) - numpy.datetime64(date)
    return td


@contextmanager
def timing(description: str) -> None:
    """Measure execution time of context.

    Examples:
        >>> with timing('Total'):
        ...     pass # logic here
    """
    start = time()
    yield
    ellapsed_time = time() - start

    print(f'{description}: {ellapsed_time} seconds')


def _common_bands():
    """Return the BDC common bands."""
    return TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME, 'cnc', DATASOURCE_NAME


def _has_default_or_index_bands(band: Band) -> bool:
    return band.common_name.lower() in ('ndvi', 'evi',) or (band.metadata_ and band.metadata_.get('expression'))


class Maestro:
    """Define class for handling data cube generation."""

    datacube: Collection = None
    reused_datacube: Collection = None
    _warped = None
    bands = None
    tiles = None
    mosaics = None
    cached_stacs = None
    band_map = None

    def __init__(self, datacube: str, collections: List[str], tiles: List[str], start_date: str, end_date: str, **properties):
        """Build Maestro interface."""
        self.params = dict(
            datacube=datacube,
            collections=collections,
            tiles=tiles,
            start_date=start_date,
            end_date=end_date
        )

        bands = properties.pop('bands', None)

        if bands:
            self.params['bands'] = bands

        force = properties.get('force', False)
        self.properties = properties
        self.params['force'] = force
        self.mosaics = dict()
        self.cached_stacs = dict()
        self.bands = []
        self.tiles = []
        self.export_files = self.properties.pop('export_files', None)

    def get_stac(self, collection: str) -> STAC:
        """Retrieve STAC client which provides the given collection.

        By default, it searches for given collection on Brazil Data Cube STAC.
        When collection not found, search on INPE STAC.

        Args:
            collection - Collection name to search

        Returns:
            STAC client
        """
        if self.properties.get('stac_url'):
            return self._stac(collection, self.properties['stac_url'], **self.properties)

        try:
            return self._stac(collection, Config.STAC_URL, **self.properties)
        except RuntimeError:
            # Search in INPE STAC
            return self._stac(collection, 'http://cdsr.dpi.inpe.br/inpe-stac/stac')

    def _stac(self, collection: str, url: str, **kwargs) -> STAC:
        """Check if collection is provided by given STAC url.

        The provided STAC must follow the `SpatioTemporal Asset Catalogs spec <https://stacspec.org/>`_.

        Exceptions:
            RuntimeError for any exception during STAC connection.

        Args:
            collection: Collection name
            url - STAC URL

        Returns:
            STAC client
        """
        try:
            options = dict()
            if kwargs.get('token'):
                options['access_token'] = kwargs.get('token')

            stac = self.cached_stacs.get(url) or STAC(url, **options)

            _ = stac.catalog

            _ = stac.collection(collection)

            self.cached_stacs.setdefault(url, stac)

            return stac
        except Exception as e:
            # STAC Error
            raise RuntimeError('An error occurred in STAC {}'.format(str(e)))

    def orchestrate(self):
        """Orchestrate datacube defintion and prepare temporal resolutions."""
        self.datacube = (
            Collection.query()
            .filter(Collection.name == self.params['datacube'])
            .first_or_404(f'Cube {self.params["datacube"]} not found.')
        )

        temporal_schema = self.datacube.temporal_composition_schema

        cube_parameters = self._check_parameters()

        # Pass the cube parameters to the data cube functions arguments
        props = deepcopy(cube_parameters.metadata_)
        props.update(self.properties)
        self.properties = props

        dstart = self.params['start_date']
        dend = self.params['end_date']

        if self.datacube.composite_function.alias == IDENTITY:
            timeline = [[dstart, dend]]
        else:
            if self.datacube.composite_function.alias == 'STK':
                warnings.warn('The composite function STK is deprecated. Use LCF (Least Cloud Cover First) instead.',
                              DeprecationWarning, stacklevel=2)

            timeline = Timeline(**temporal_schema, start_date=dstart, end_date=dend).mount()

        where = [Tile.grid_ref_sys_id == self.datacube.grid_ref_sys_id]

        self.bands = Band.query().filter(Band.collection_id == self.warped_datacube.id).all()

        bands = self.datacube_bands
        self.band_map = {b.name: dict(name=b.name, data_type=b.data_type, nodata=b.nodata,
                                      min_value=b.min_value, max_value=b.max_value) for b in bands}

        if self.properties.get('reuse_from'):
            warnings.warn(
                'The parameter `reuse_from` is deprecated and will be removed in next version. '
                'Use `reuse_data_cube` instead.'
            )
            common_bands = _common_bands()
            collection_bands = [b.name for b in self.datacube.bands if b.name not in common_bands]

            reused_collection_bands = [b.name for b in self.bands]

            # The input cube (STK/MED) must have all bands of reused. Otherwise raise Error.
            if not set(collection_bands).issubset(set(reused_collection_bands)):
                raise RuntimeError(
                    f'Reused data cube {self.warped_datacube.name} must have all bands of {self.datacube.name}'
                )

            # Extra filter to only use bands of Input data cube.
            self.bands = [b for b in self.bands if b.name in collection_bands]

        if cube_parameters.reuse_cube:
            self.reused_datacube = cube_parameters.reuse_cube

        quality_band = None
        if self.properties.get('quality_band'):
            quality_band = self.properties['quality_band']

            quality = next(filter(lambda b: b.name == quality_band, bands))
            self.properties['mask']['nodata'] = float(quality.nodata)
        else:
            self.properties['quality_band'] = None
            self.properties['mask'] = None

        # When reuse Identity data cubes, set both name/version as identifier
        if self.reused_datacube:
            self.properties['reuse_data_cube'] = dict(
                name=self.reused_datacube.name,
                version=self.reused_datacube.version,
            )

        if not self.params.get('tiles'):
            abort(400, 'Missing tiles in data cube generation')

        where.append(Tile.name.in_(self.params['tiles']))
        tiles = db.session.query(Tile).filter(*where).all()

        if len(tiles) != len(self.params['tiles']):
            tiles_not_found = set(self.params['tiles']) - set([t.name for t in tiles])
            abort(400, f'Tiles not found {tiles_not_found}')

        self.tiles = tiles

        for tile in self.tiles:
            tile_name = tile.name

            grs: GridRefSys = tile.grs

            grid_geom: sqlalchemy.Table = grs.geom_table

            srid_column = get_srid_column(grid_geom.c)

            # TODO: Raise exception when using a native grid argument
            #  Use bands resolution and match with SRID context (degree x degree) etc.

            tile_stats = db.session.query(
                (func.ST_XMin(grid_geom.c.geom)).label('min_x'),
                (func.ST_YMax(grid_geom.c.geom)).label('max_y'),
                (func.ST_XMax(grid_geom.c.geom) - func.ST_XMin(grid_geom.c.geom)).label('dist_x'),
                (func.ST_YMax(grid_geom.c.geom) - func.ST_YMin(grid_geom.c.geom)).label('dist_y'),
                (func.ST_Transform(
                    func.ST_SetSRID(grid_geom.c.geom, srid_column), 4326
                )).label('feature'),
                grid_geom.c.geom.label('geom')
            ).filter(grid_geom.c.tile == tile_name).first()

            self.mosaics[tile_name] = dict(
                periods=dict()
            )

            for interval in timeline:
                startdate = interval[0]
                enddate = interval[1]

                if dstart is not None and startdate < dstart:
                    continue
                if dend is not None and enddate > dend:
                    continue

                period = f'{startdate}_{enddate}'

                self.mosaics[tile_name]['periods'][period] = {}
                self.mosaics[tile_name]['periods'][period]['start'] = startdate.strftime('%Y-%m-%d')
                self.mosaics[tile_name]['periods'][period]['end'] = enddate.strftime('%Y-%m-%d')
                self.mosaics[tile_name]['periods'][period]['dist_x'] = tile_stats.dist_x
                self.mosaics[tile_name]['periods'][period]['dist_y'] = tile_stats.dist_y
                self.mosaics[tile_name]['periods'][period]['min_x'] = tile_stats.min_x
                self.mosaics[tile_name]['periods'][period]['max_y'] = tile_stats.max_y
                self.mosaics[tile_name]['periods'][period]['feature'] = to_shape(tile_stats.feature)
                self.mosaics[tile_name]['periods'][period]['geom'] = to_shape(tile_stats.geom)
                if self.properties.get('shape', None):
                    self.mosaics[tile_name]['periods'][period]['shape'] = self.properties['shape']

    def _check_parameters(self) -> CubeParameters:
        cube_parameters: CubeParameters = CubeParameters.query().filter(
            CubeParameters.collection_id == self.datacube.id
        ).first()

        if cube_parameters is None:
            raise RuntimeError(f'No parameters configured for data cube "{self.datacube.id}"')

        # This step acts like first execution.
        # When no stac_url/local defined in cube parameters, save it initial from parameters.
        if self.properties.get('local') and not cube_parameters.metadata_.get('local'):
            pass
        elif self.properties.get('stac_url') and not cube_parameters.metadata_.get('stac_url'):
            logging.debug(f'No "stac_url"/"token" configured yet for cube parameters.'
                          f'Using {self.properties["stac_url"]}')
            meta = cube_parameters.metadata_.copy()
            meta['stac_url'] = self.properties['stac_url']
            meta['token'] = self.properties.get('token')
            cube_parameters.metadata_ = meta
            cube_parameters.save(commit=True)

        # Validate parameters
        cube_parameters.validate()

        if self.properties.get('stac_url') and self.params.get('collections') is None:
            raise RuntimeError(f'Missing parameter "collections" for STAC {self.properties.get("stac_url")}')

        return cube_parameters

    @property
    def warped_datacube(self) -> Collection:
        """Retrieve cached datacube definition."""
        if not self._warped:
            if self.properties.get('reuse_from'):
                reused_datacube: Collection = Collection.query().filter(
                    Collection.name == self.properties['reuse_from']).first()

                if reused_datacube is None:
                    raise RuntimeError(f'Data cube {self.properties["reuse_from"]} not found.')

                if reused_datacube.composite_function.alias != IDENTITY:
                    raise RuntimeError(f'Data cube {self.properties["reuse_from"]} must be {IDENTITY}.')

                if reused_datacube.grid_ref_sys_id != self.datacube.grid_ref_sys_id:
                    raise RuntimeError(
                        f'The grid of data cube {self.datacube.name} and {reused_datacube.name} mismatch.')

                self.reused_datacube = reused_datacube
                # set warped_collection to reused

                if self.params['force']:
                    raise RuntimeError(
                        f'Cannot use flag --force to dispatch data cube derived from {reused_datacube.name}')

                self._warped = reused_datacube
            else:
                source = self.datacube
                if self.datacube.composite_function.alias != 'IDT':
                    source = self.source(self.datacube, composite_function='IDT')
                    if source is None:
                        raise RuntimeError(f'Missing Identity cube for {self.datacube.name}-{self.datacube.version}')

                self._warped = source

        return self._warped

    @staticmethod
    def sources(cube: Collection) -> List[Collection]:
        """Trace data cube collection origin.

        It traces all the collection origin from the given datacube using
        ``bdc_catalog.models.CollectionSRC``.
        """
        out = []
        ref = cube
        while ref is not None:
            source: CollectionSRC = (
                CollectionSRC.query()
                .filter(CollectionSRC.collection_id == ref.id)
                .first()
            )
            if source is None:
                break

            ref = Collection.query().get(source.collection_src_id)
            out.append(ref)
        return out

    @staticmethod
    def source(cube: Collection, composite_function=None):
        """Trace the first data cube origin.

        Args:
            cube (Collection): Data cube to trace.
            composite_function (str): String composite function to filter.
        """
        sources = Maestro.sources(cube)
        for collection in sources:
            if (composite_function and collection.composite_function and
                    collection.composite_function.alias == composite_function):
                return collection

    @property
    def datacube_bands(self) -> List[Band]:
        """Retrieve data cube bands based int user input."""
        if self.params.get('bands'):
            return list(filter(lambda band: band.name in self.params['bands'], self.bands))
        return [b for b in self.bands if b.name != DATASOURCE_NAME]

    def dispatch_celery(self):
        """Dispatch datacube generation on celery workers.

        Make sure celery is running. Check RUNNING.rst for further details.
        """
        with timing('Time total to dispatch'):
            bands = self.datacube_bands
            common_bands = _common_bands()
            export = dict()
            output = dict(merges=dict(), blends=dict())
            stac_kwargs = self.properties.get('stac_kwargs', dict())
            local = self.properties.get('local')
            fmt = self.properties.get('format')
            pattern = self.properties.get('pattern')
            local_resolver = None
            if local and fmt:
                local_resolver = load_format(fmt)

            band_str_list = [
                band.name for band in bands if band.name not in common_bands
            ]

            warped_datacube = self.warped_datacube.name

            for tileid in self.mosaics:
                blends = []
                export.setdefault(tileid, dict())

                tile = next(filter(lambda t: t.name == tileid, self.tiles))

                grid_crs = tile.grs.crs

                # For each blend
                for period in self.mosaics[tileid]['periods']:
                    start = self.mosaics[tileid]['periods'][period]['start']
                    end = self.mosaics[tileid]['periods'][period]['end']

                    output['merges'].setdefault(period, dict())

                    feature = self.mosaics[tileid]['periods'][period]['feature']

                    if local and fmt:
                        start = datetime.datetime.strptime(start, '%Y-%m-%d')
                        end = datetime.datetime.strptime(end, '%Y-%m-%d')
                        files = local_resolver.files(local, pattern=pattern, recursive=True)
                        tile_geom = self.mosaics[tileid]['periods'][period]['geom']
                        assets_by_period = local_resolver.create_collection(tile_geom,
                                                                            grid_crs,
                                                                            files=files,
                                                                            start=start,
                                                                            end=end)
                    else:
                        assets_by_period = self.search_images(shapely.geometry.mapping(feature), start, end, tileid, **stac_kwargs)

                    if self.datacube.composite_function.alias == IDENTITY:
                        stats_bands = [TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME, DATASOURCE_NAME]

                        stats_bands.extend([b.name for b in bands if b.name not in stats_bands and _has_default_or_index_bands(b)])
                        # Mount list of True/False values
                        is_any_empty = list(map(lambda k: k not in stats_bands and len(assets_by_period[k]) == 0, assets_by_period.keys()))
                        # When no asset found in this period, skip it.
                        if any(is_any_empty):
                            continue

                    merges_tasks = []

                    dist_x = self.mosaics[tileid]['periods'][period]['dist_x']
                    dist_y = self.mosaics[tileid]['periods'][period]['dist_y']
                    min_x = self.mosaics[tileid]['periods'][period]['min_x']
                    max_y = self.mosaics[tileid]['periods'][period]['max_y']
                    start_date = self.mosaics[tileid]['periods'][period]['start']
                    end_date = self.mosaics[tileid]['periods'][period]['end']
                    period_start_end = '{}_{}'.format(start_date, end_date)
                    export[tileid].setdefault(period_start_end, dict())
                    merge_opts = dict()

                    for band in bands:
                        # Skip trigger/search for Vegetation Index
                        if _has_default_or_index_bands(band):
                            continue

                        export[tileid][period_start_end].setdefault(band.name, [])
                        merges = assets_by_period[band.name]

                        resolutions = band.eo_resolutions or self.params.get('resolutions')[band.name]

                        if not merges:
                            for _b in bands:
                                if _b.name == band.name or _has_default_or_index_bands(_b):
                                    continue
                                _m = assets_by_period[_b.name]
                                if _m:
                                    raise RuntimeError(
                                        f'Unexpected Error: The band {_b.name} has scenes, however '
                                        f'there any bands ({band.name}) that don\'t have any scenes on provider.'
                                    )
                                _m[start_date] = dict()

                            # Adapt to make the merge function to generate empty raster
                            merges[start_date] = dict()
                            merge_opts['empty'] = True

                        for merge_date, collections in merges.items():
                            assets = []
                            # Preserve collections order
                            for values in collections.values():
                                assets.extend(values)

                            export[tileid][period_start_end][band.name].extend(assets)
                            output['merges'][period].setdefault(merge_date, [])

                            properties = dict(
                                date=merge_date,
                                datasets=collections,
                                xmin=min_x,
                                ymax=max_y,
                                resx=float(resolutions[0]),
                                resy=float(resolutions[1]),
                                dist_x=dist_x,
                                dist_y=dist_y,
                                srs=grid_crs,
                                tile_id=tileid,
                                assets=assets,
                                nodata=float(band.nodata),
                                bands=band_str_list,
                                version=self.datacube.version,
                                **merge_opts
                            )

                            activity = get_or_create_activity(
                                cube=self.datacube.name,
                                warped=warped_datacube,
                                activity_type='MERGE',
                                scene_type='WARPED',
                                band=band.name,
                                period=period_start_end,
                                activity_date=merge_date,
                                **properties
                            )

                            output['merges'][period][merge_date].append(activity)

                            task = warp_merge.s(activity, self.band_map, **self.properties)
                            merges_tasks.append(task)

                    if len(merges_tasks) > 0:
                        task = chain(group(merges_tasks), prepare_blend.s(self.band_map, **self.properties))
                        blends.append(task)
                        output['blends'][period] = task

                if len(blends) > 0:
                    task = group(blends)
                    task.apply_async()

            if self.export_files:
                file_path = Path(self.export_files)
                for tile, period_map in export.items():
                    for period, assets in period_map.items():
                        file_path_period = file_path.parent / f'{file_path.stem}-{tile}-{period}.json'
                        file_path_period.parent.mkdir(exist_ok=True, parents=True)
                        with open(file_path_period, 'w') as f:
                            f.write(json.dumps(assets, indent=4))

        return output

    def search_images(self, feature: dict, start: str, end: str, tile_id: str, **kwargs):
        """Search and prepare images on STAC."""
        scenes = {}
        options = dict(
            intersects=feature,
            datetime='{}T00:00:00/{}T23:59:59'.format(start, end),
            limit=1000
        )
        options.update(kwargs)

        bands = self.datacube_bands

        # Retrieve band definition in dict format.
        # TODO: Should we use from STAC?
        collection_bands = dict()

        for band_obj in bands:
            collection_bands[band_obj.name] = dict(
                min_value=float(band_obj.min_value),
                max_value=float(band_obj.max_value),
                nodata=float(band_obj.nodata),
            )

        for band in bands:
            if band.name != PROVENANCE_NAME:
                scenes[band.name] = dict()

        for dataset in self.params['collections']:
            options['collections'] = [dataset]
            stac = self.get_stac(dataset)

            token = ''

            print('Searching for {} - {} ({}, {}) using {}...'.format(dataset, tile_id, start,
                                                                      end, stac.url), end='', flush=True)

            with timing(' total'):
                items = stac.search(filter=options)

                for feature in items['features']:
                    if feature['type'] == 'Feature':
                        date = feature['properties']['datetime'][0:10]
                        identifier = feature['id']
                        stac_bands = feature['properties'].get('eo:bands', [])

                        for band in bands:
                            band_name_href = band.name
                            if 'CBERS' in dataset and band.common_name not in ('evi', 'ndvi'):
                                band_name_href = band.common_name

                            elif band.name not in feature['assets']:
                                if f'sr_{band.name}' not in feature['assets']:
                                    continue
                                else:
                                    band_name_href = f'sr_{band.name}'

                            feature_band = list(filter(lambda b: b['name'] == band_name_href,stac_bands))
                            feature_band = feature_band[0] if len(feature_band) > 0 else dict()

                            scenes[band.name].setdefault(date, dict())

                            link = feature['assets'][band_name_href]['href']

                            scene = dict(**collection_bands[band.name])
                            scene.update(**feature_band)

                            scene['sceneid'] = identifier
                            scene['band'] = band.name
                            scene['dataset'] = dataset

                            link = link.replace(Config.CBERS_SOURCE_URL_PREFIX, Config.CBERS_TARGET_URL_PREFIX)

                            if token:
                                link = '{}{}'.format(link, token)

                            scene['link'] = link

                            if dataset == 'MOD13Q1' and band.common_name == 'quality':
                                scene['link'] = scene['link'].replace('quality', 'reliability')

                            scenes[band.name][date].setdefault(dataset, [])
                            scenes[band.name][date][dataset].append(scene)

        return scenes
