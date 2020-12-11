#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder forms used to validate both data input and data serialization."""

# Python
import datetime
from contextlib import contextmanager
from time import time
from typing import List

# 3rdparty
import numpy
from bdc_catalog.models import Band, Collection, GridRefSys, Tile, db
from celery import chain, group
from geoalchemy2 import func
from stac import STAC

# Cube Builder
from .celery.tasks import prepare_blend, warp_merge
from .config import Config
from .constants import (CLEAR_OBSERVATION_NAME, DATASOURCE_NAME,
                        PROVENANCE_NAME, TOTAL_OBSERVATION_NAME)
from .utils.processing import get_cube_id, get_or_create_activity
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


class Maestro:
    """Define class for handling data cube generation."""

    datacube: Collection = None
    reused_datacube: Collection = None
    _warped = None
    bands = None
    tiles = None
    mosaics = None
    cached_stacs = None

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
        self.datacube = Collection.query().filter(Collection.name == self.params['datacube']).one()

        temporal_schema = self.datacube.temporal_composition_schema

        dstart = self.params['start_date']
        dend = self.params['end_date']

        timeline = Timeline(**temporal_schema, start_date=dstart, end_date=dend).mount()

        where = [Tile.grid_ref_sys_id == self.datacube.grid_ref_sys_id]

        if self.params.get('tiles'):
            where.append(Tile.name.in_(self.params['tiles']))

        self.tiles = db.session.query(Tile).filter(*where).all()

        self.bands = Band.query().filter(Band.collection_id == self.warped_datacube.id).all()

        if self.properties.get('reuse_from'):
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

        for tile in self.tiles:
            tile_name = tile.name

            grs: GridRefSys = tile.grs

            grid_geom = grs.geom_table

            tile_stats = db.session.query(
                (func.ST_XMin(grid_geom.c.geom)).label('min_x'),
                (func.ST_YMax(grid_geom.c.geom)).label('max_y'),
                (func.ST_XMax(grid_geom.c.geom) - func.ST_XMin(grid_geom.c.geom)).label('dist_x'),
                (func.ST_YMax(grid_geom.c.geom) - func.ST_YMin(grid_geom.c.geom)).label('dist_y')
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
                cube_relative_path = f'{self.datacube.name}/v{self.datacube.version:03d}/{tile_name}/{period}'

                self.mosaics[tile_name]['periods'][period] = {}
                self.mosaics[tile_name]['periods'][period]['start'] = startdate.strftime('%Y-%m-%d')
                self.mosaics[tile_name]['periods'][period]['end'] = enddate.strftime('%Y-%m-%d')
                self.mosaics[tile_name]['periods'][period]['dist_x'] = tile_stats.dist_x
                self.mosaics[tile_name]['periods'][period]['dist_y'] = tile_stats.dist_y
                self.mosaics[tile_name]['periods'][period]['min_x'] = tile_stats.min_x
                self.mosaics[tile_name]['periods'][period]['max_y'] = tile_stats.max_y
                self.mosaics[tile_name]['periods'][period]['dirname'] = cube_relative_path
                if self.properties.get('shape', None):
                    self.mosaics[tile_name]['periods'][period]['shape'] = self.properties['shape']

    @property
    def warped_datacube(self) -> Collection:
        """Retrieve cached datacube defintion."""
        if not self._warped:
            if self.properties.get('reuse_from'):
                reused_datacube: Collection = Collection.query().filter(
                    Collection.name == self.properties['reuse_from']).first()

                if reused_datacube is None:
                    raise RuntimeError(f'Data cube {self.properties["reuse_from"]} not found.')

                if reused_datacube.composite_function.alias != 'IDT':
                    raise RuntimeError(f'Data cube {self.properties["reuse_from"]} must be IDT.')

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
                datacube_warped = get_cube_id(self.datacube.name)

                self._warped = Collection.query().filter(Collection.name == datacube_warped).first()

        return self._warped

    @property
    def datacube_bands(self) -> List[Band]:
        """Retrieve data cube bands based int user input."""
        if self.params.get('bands'):
            return list(filter(lambda band: band.name in self.params['bands'], self.bands))
        return [b for b in self.bands if b.name != DATASOURCE_NAME]

    @staticmethod
    def get_bbox(tile_id: str, grs: GridRefSys) -> str:
        """Retrieve the bounding box representation as string."""
        geom_table = grs.geom_table

        bbox_result = db.session.query(
            geom_table.c.tile,
            func.ST_AsText(func.ST_BoundingDiagonal(func.ST_Transform(geom_table.c.geom, 4326)))
        ).filter(
            geom_table.c.tile == tile_id
        ).first()

        bbox = bbox_result[1][bbox_result[1].find('(') + 1:bbox_result[0].find(')')]
        bbox = bbox.replace(' ', ',')

        return bbox

    def dispatch_celery(self):
        """Dispatch datacube generation on celery workers.

        Make sure celery is running. Check RUNNING.rst for further details.
        """
        with timing('Time total to dispatch'):
            bands = self.datacube_bands
            common_bands = _common_bands()

            band_str_list = [
                band.name for band in bands if band.name not in common_bands
            ]

            band_map = {b.common_name: b.name for b in bands}

            if not band_map.get('quality'):
                raise RuntimeError('Quality band is required')

            warped_datacube = self.warped_datacube.name

            for tileid in self.mosaics:
                blends = []

                bbox = self.get_bbox(tileid, self.datacube.grs)

                tile = next(filter(lambda t: t.name == tileid, self.tiles))

                grid_crs = tile.grs.crs

                # For each blend
                for period in self.mosaics[tileid]['periods']:
                    start = self.mosaics[tileid]['periods'][period]['start']
                    end = self.mosaics[tileid]['periods'][period]['end']

                    assets_by_period = self.search_images(bbox, start, end, tileid)

                    if self.datacube.composite_function.alias == 'IDT':
                        stats_bands = (TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME, DATASOURCE_NAME)
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

                    for band in bands:
                        # Skip trigger/search for Vegetation Index
                        if band.common_name.lower() in ('ndvi', 'evi',):
                            continue

                        merges = assets_by_period[band.name]

                        for merge_date, collections in merges.items():
                            assets = []
                            # Preserve collections order
                            for values in collections.values():
                                assets.extend(values)

                            properties = dict(
                                date=merge_date,
                                datasets=self.params['collections'],
                                xmin=min_x,
                                ymax=max_y,
                                resx=float(band.resolution_x),
                                resy=float(band.resolution_y),
                                dist_x=dist_x,
                                dist_y=dist_y,
                                srs=grid_crs,
                                tile_id=tileid,
                                assets=assets,
                                nodata=float(band.nodata),
                                bands=band_str_list,
                                version=self.datacube.version
                            )

                            if self.reused_datacube:
                                properties['reuse_datacube'] = self.reused_datacube.id

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

                            task = warp_merge.s(activity, band_map, **self.properties)
                            merges_tasks.append(task)

                    if len(merges_tasks) > 0:
                        task = chain(group(merges_tasks), prepare_blend.s(band_map, **self.properties))
                        blends.append(task)

                if len(blends) > 0:
                    task = group(blends)
                    task.apply_async()

        return self.mosaics

    def search_images(self, bbox: str, start: str, end: str, tile_id: str):
        """Search and prepare images on STAC."""
        scenes = {}
        options = dict(
            bbox=bbox,
            datetime='{}/{}'.format(start, end),
            limit=100000
        )

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
            if 'CBERS' in dataset:
                options = dict(
                    bbox=bbox,
                    time='{}/{}'.format(start, end),
                    limit=100000
                )
            stac = self.get_stac(dataset)

            token = ''

            print('Searching for {} - {} ({}, {}) using {}...'.format(dataset, tile_id, start,
                                                                      end, stac.url), end='', flush=True)

            with timing(' total'):

                if 'CBERS' in dataset and Config.CBERS_AUTH_TOKEN:
                    token = '?key={}'.format(Config.CBERS_AUTH_TOKEN)

                items = stac.collection(dataset).get_items(filter=options)

                for feature in items['features']:
                    if feature['type'] == 'Feature':
                        date = feature['properties']['datetime'][0:10]
                        identifier = feature['id']

                        for band in bands:
                            band_name_href = band.name
                            if 'CBERS' in dataset and band.common_name not in ('evi', 'ndvi'):
                                band_name_href = band.common_name

                            elif band.name not in feature['assets']:
                                if f'sr_{band.name}' not in feature['assets']:
                                    continue
                                else:
                                    band_name_href = f'sr_{band.name}'

                            scenes[band.name].setdefault(date, dict())

                            link = feature['assets'][band_name_href]['href']

                            scene = dict(**collection_bands[band.name])
                            scene['sceneid'] = identifier
                            scene['band'] = band.name
                            scene['dataset'] = dataset

                            link = link.replace('cdsr.dpi.inpe.br/api/download/TIFF', 'www.dpi.inpe.br/catalog/tmp')

                            if token:
                                link = '{}{}'.format(link, token)

                            scene['link'] = link

                            if dataset == 'MOD13Q1' and band.common_name == 'quality':
                                scene['link'] = scene['link'].replace('quality', 'reliability')

                            scenes[band.name][date].setdefault(dataset, [])
                            scenes[band.name][date][dataset].append(scene)

        return scenes
