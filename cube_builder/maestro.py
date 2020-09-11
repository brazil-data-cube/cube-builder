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
from typing import List, Union, Optional

# 3rdparty
import numpy
from bdc_catalog.models import Band, Collection, Tile, db, GridRefSys
from celery import group, chain
from dateutil.relativedelta import relativedelta
from geoalchemy2 import func
from stac import STAC

# Cube Builder
from .celery.tasks import prepare_blend, warp_merge
from .config import Config
from .constants import TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME
from .utils import get_cube_id, get_or_create_activity


SchemaType = Union['cyclic', 'continuous']


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


def decode_periods(schema: SchemaType, step: int, unit: str, start_date, end_date, cycle: Optional[dict] = None):
    """Retrieve datacube temporal resolution by periods.

    TODO: Describe how it works.
    """
    requested_periods = {}
    if start_date is None:
        return requested_periods
    if isinstance(start_date, datetime.date):
        start_date = start_date.strftime('%Y-%m-%d')

    step = int(step)

    td_time_step = datetime.timedelta(days=step)
    steps_per_period = int(round(365./step))

    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if isinstance(end_date, datetime.date):
        end_date = end_date.strftime('%Y-%m-%d')

    if unit == 'M' and schema == 'continuous':
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        delta = relativedelta(months=step)
        requested_period = []
        while start_date <= end_date:
            next_date = start_date + delta
            periodkey = str(start_date)[:10] + '_' + str(start_date)[:10] + '_' + str(next_date - relativedelta(days=1))[:10]
            requested_period.append(periodkey)
            requested_periods[start_date] = requested_period
            start_date = next_date
        return requested_periods

    # Find the exact start_date based on periods that start on yyyy-01-01
    firstyear = start_date.split('-')[0]
    new_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if unit == 'day':
        dbase = datetime.datetime.strptime(firstyear+'-01-01', '%Y-%m-%d')
        while dbase < new_start_date:
            dbase += td_time_step
        if dbase > new_start_date:
            dbase -= td_time_step
        start_date = dbase.strftime('%Y-%m-%d')
        new_start_date = dbase

    # Find the exact end_date based on periods that start on yyyy-01-01
    lastyear = end_date.split('-')[0]
    new_end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    if unit == 'day':
        dbase = datetime.datetime.strptime(lastyear+'-12-31', '%Y-%m-%d')
        while dbase > new_end_date:
            dbase -= td_time_step
        end_date = dbase
        if end_date == start_date:
            end_date += td_time_step - datetime.timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')

    # For annual periods
    if unit == 'day' and cycle:
        dbase = new_start_date
        yearold = dbase.year
        count = 0
        requested_period = []
        while dbase < new_end_date:
            if yearold != dbase.year:
                dbase = datetime.datetime(dbase.year,1,1)
            yearold = dbase.year
            dstart = dbase
            dend = dbase + td_time_step - datetime.timedelta(days=1)
            dend = min(datetime.datetime(dbase.year, 12, 31), dend)
            basedate = dbase.strftime('%Y-%m-%d')
            start_date = dstart.strftime('%Y-%m-%d')
            end_date = dend.strftime('%Y-%m-%d')
            periodkey = basedate + '_' + start_date + '_' + end_date
            if count % steps_per_period == 0:
                count = 0
                requested_period = []
                requested_periods[basedate] = requested_period
            requested_period.append(periodkey)
            count += 1
            dbase += td_time_step
        if len(requested_periods) == 0 and count > 0:
            requested_periods[basedate].append(requested_period)
    else:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        yeari = start_date.year
        yearf = end_date.year
        monthi = start_date.month
        monthf = end_date.month
        dayi = start_date.day
        dayf = end_date.day
        for year in range(yeari,yearf+1):
            dbase = datetime.datetime(year,monthi,dayi)
            if monthi <= monthf:
                dbasen = datetime.datetime(year,monthf,dayf)
            else:
                dbasen = datetime.datetime(year+1,monthf,dayf)
            while dbase < dbasen:
                dstart = dbase
                dend = dbase + td_time_step - datetime.timedelta(days=1)
                basedate = dbase.strftime('%Y-%m-%d')
                start_date = dstart.strftime('%Y-%m-%d')
                end_date = dend.strftime('%Y-%m-%d')
                periodkey = basedate + '_' + start_date + '_' + end_date
                requested_period = []
                requested_periods[basedate] = requested_period
                requested_periods[basedate].append(periodkey)
                dbase += td_time_step
    return requested_periods


class Maestro:
    """Define class for handling data cube generation."""

    datacube: Collection = None
    _warped = None
    bands = []
    tiles = []
    mosaics = dict()
    cached_stacs = dict()

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

    def get_stac(self, collection: str) -> STAC:
        """Retrieve STAC client which provides the given collection.

        By default, it searches for given collection on Brazil Data Cube STAC.
        When collection not found, search on INPE STAC.

        Args:
            collection - Collection name to search

        Returns:
            STAC client
        """
        try:
            return self._stac(collection, Config.STAC_URL, **self.properties)
        except RuntimeError:
            # Search in INPE STAC
            return self._stac(collection, 'http://cdsr.dpi.inpe.br/inpe-stac')

    @classmethod
    def _stac(cls, collection: str, url: str, **kwargs) -> STAC:
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

            stac = cls.cached_stacs.get(url) or STAC(url, **options)

            _ = stac.catalog

            _ = stac.collection(collection)

            cls.cached_stacs.setdefault(url, stac)

            return stac
        except Exception as e:
            # STAC Error
            raise RuntimeError('An error occurred in STAC {}'.format(str(e)))

    def orchestrate(self):
        """Orchestrate datacube defintion and prepare temporal resolutions."""
        self.datacube = Collection.query().filter(Collection.name == self.params['datacube']).one()

        temporal_schema = self.datacube.temporal_composition_schema

        cube_start_date = self.params['start_date']

        dstart = self.params['start_date']
        dend = self.params['end_date']

        if cube_start_date is None:
            cube_start_date = dstart.strftime('%Y-%m-%d')

        cube_end_date = dend.strftime('%Y-%m-%d')

        periodlist = decode_periods(**temporal_schema, start_date=cube_start_date, end_date=cube_end_date)

        where = [Tile.grid_ref_sys_id == self.datacube.grid_ref_sys_id]

        if self.params.get('tiles'):
            where.append(Tile.name.in_(self.params['tiles']))

        self.tiles = db.session.query(Tile).filter(*where).all()

        self.bands = Band.query().filter(Band.collection_id == self.warped_datacube.id).all()

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

            for datekey in sorted(periodlist):
                requested_period = periodlist[datekey]
                for periodkey in requested_period:
                    _, startdate, enddate = periodkey.split('_')

                    if dstart is not None and startdate < dstart.strftime('%Y-%m-%d'):
                        continue
                    if dend is not None and enddate > dend.strftime('%Y-%m-%d'):
                        continue

                    period = f'{startdate}_{enddate}'
                    cube_relative_path = '{0}/v{1:03d}/{2}/{3}'.format(self.datacube.name, self.datacube.version,
                                                                       tile_name, period)

                    self.mosaics[tile_name]['periods'][periodkey] = {}
                    self.mosaics[tile_name]['periods'][periodkey]['start'] = startdate
                    self.mosaics[tile_name]['periods'][periodkey]['end'] = enddate
                    self.mosaics[tile_name]['periods'][periodkey]['dist_x'] = tile_stats.dist_x
                    self.mosaics[tile_name]['periods'][periodkey]['dist_y'] = tile_stats.dist_y
                    self.mosaics[tile_name]['periods'][periodkey]['min_x'] = tile_stats.min_x
                    self.mosaics[tile_name]['periods'][periodkey]['max_y'] = tile_stats.max_y
                    self.mosaics[tile_name]['periods'][periodkey]['dirname'] = cube_relative_path

    @property
    def warped_datacube(self) -> Collection:
        """Retrieve cached datacube defintion."""
        if not self._warped:
            datacube_warped = get_cube_id(self.datacube.name)

            self._warped = Collection.query().filter(Collection.name == datacube_warped).first()

        return self._warped

    @property
    def datacube_bands(self) -> List[Band]:
        """Retrieve data cube bands based int user input."""
        if self.params.get('bands'):
            return list(filter(lambda band: band.name in self.params['bands'], self.bands))
        return self.bands

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

            band_str_list = [band.name for band in bands if band.name not in [TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME, 'cnc']]

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
                        stats_bands = (TOTAL_OBSERVATION_NAME, CLEAR_OBSERVATION_NAME, PROVENANCE_NAME,)
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

                        collections = assets_by_period[band.name]

                        for collection, merges in collections.items():
                            for merge_date, assets in merges.items():
                                properties = dict(
                                    date=merge_date,
                                    dataset=collection,
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
            scenes[band.name] = dict()

        for dataset in self.params['collections']:
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
                            if band.name not in feature['assets']:
                                continue

                            scenes[band.name].setdefault(dataset, dict())

                            link = feature['assets'][band.name]['href']

                            scene = dict(**collection_bands[band.name])
                            scene['sceneid'] = identifier
                            scene['band'] = band.name

                            link = link.replace('cdsr.dpi.inpe.br/api/download/TIFF', 'www.dpi.inpe.br/catalog/tmp')

                            if token:
                                link = '{}{}'.format(link, token)

                            scene['link'] = link

                            if dataset == 'MOD13Q1' and band.common_name == 'quality':
                                scene['link'] = scene['link'].replace('quality', 'reliability')

                            scenes[band.name][dataset].setdefault(date, [])
                            scenes[band.name][dataset][date].append(scene)

        return scenes
