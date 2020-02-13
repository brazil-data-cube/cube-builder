# Python
from typing import List
import datetime

# 3rdparty
from bdc_db.models import Collection, Tile, Band, db, CollectionItem
from geoalchemy2 import func
from stac import STAC
import numpy

# BDC Scripts
from .config import Config
from .models.activity import Activity


def days_in_month(date):
    year = int(date.split('-')[0])
    month = int(date.split('-')[1])
    nday = day = int(date.split('-')[2])
    if month == 12:
        nmonth = 1
        nyear = year + 1
    else:
        nmonth = month + 1
        nyear = year
    ndate = '{0:4d}-{1:02d}-{2:02d}'.format(nyear,nmonth,nday)
    td = numpy.datetime64(ndate) - numpy.datetime64(date)
    return td


def decode_periods(temporal_schema, start_date, end_date, time_step):
    print('decode_periods - {} {} {} {}'.format(temporal_schema,start_date, end_date, time_step))
    requested_periods = {}
    if start_date is None:
        return requested_periods
    if isinstance(start_date, datetime.date):
        start_date = start_date.strftime('%Y-%m-%d')

    td_time_step = datetime.timedelta(days=time_step)
    steps_per_period = int(round(365./time_step))

    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if isinstance(end_date, datetime.date):
        end_date = end_date.strftime('%Y-%m-%d')

    if temporal_schema is None:
        periodkey = start_date + '_' + start_date + '_' + end_date
        requested_period = list()
        requested_period.append(periodkey)
        requested_periods[start_date] = requested_period
        return requested_periods

    if temporal_schema == 'M':
        start_date = numpy.datetime64(start_date)
        end_date = numpy.datetime64(end_date)
        requested_period = []
        while start_date <= end_date:
            next_date = start_date + days_in_month(str(start_date))
            periodkey = str(start_date)[:10] + '_' + str(start_date)[:10] + '_' + str(next_date - numpy.timedelta64(1, 'D'))[:10]
            requested_period.append(periodkey)
            requested_periods[start_date] = requested_period
            start_date = next_date
        return requested_periods

    # Find the exact start_date based on periods that start on yyyy-01-01
    firstyear = start_date.split('-')[0]
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if temporal_schema == 'A':
        dbase = datetime.datetime.strptime(firstyear+'-01-01', '%Y-%m-%d')
        while dbase < start_date:
            dbase += td_time_step
        if dbase > start_date:
            dbase -= td_time_step
        start_date = dbase.strftime('%Y-%m-%d')
        start_date = dbase

    # Find the exact end_date based on periods that start on yyyy-01-01
    lastyear = end_date.split('-')[0]
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    if temporal_schema == 'A':
        dbase = datetime.datetime.strptime(lastyear+'-12-31', '%Y-%m-%d')
        while dbase > end_date:
            dbase -= td_time_step
        end_date = dbase
        if end_date == start_date:
            end_date += td_time_step - datetime.timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')

    # For annual periods
    if temporal_schema == 'A':
        dbase = start_date
        yearold = dbase.year
        count = 0
        requested_period = []
        while dbase < end_date:
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


stac_cli = STAC(Config.STAC_URL)


class Maestro:
    datacube = None
    bands = []
    tiles = []
    mosaics = dict()

    def __init__(self, datacube: str, collections: List[str], tiles: List[str], start_date: str, end_date: str, bands: List[str]=None):
        self.params = dict(
            datacube=datacube,
            collections=collections,
            tiles=tiles,
            start_date=start_date,
            end_date=end_date
        )

        if bands:
            self.params['bands'] = bands

    def orchestrate(self):
        self.datacube = Collection.query().filter(Collection.id == self.params['datacube']).one()

        temporal_schema = self.datacube.temporal_composition_schema.temporal_schema
        temporal_step = self.datacube.temporal_composition_schema.temporal_composite_t

        # TODO: Check in STAC for cube item
        # datacube_stac = stac_cli.collection(self.datacube.id)

        collections_items = CollectionItem.query().filter(
            CollectionItem.collection_id == self.datacube.id,
            CollectionItem.grs_schema_id == self.datacube.grs_schema_id
        ).order_by(CollectionItem.composite_start).all()
        cube_start_date = self.params['start_date']
        if list(filter(lambda c_i: c_i.tile_id == self.params['tiles'][0], collections_items)):
            cube_start_date = collections_items[0].composite_start

        dstart = self.params['start_date']
        dend = self.params['end_date']

        if cube_start_date is None:
            cube_start_date = dstart.strftime('%Y-%m-%d')

        cube_end_date = dend.strftime('%Y-%m-%d')

        periodlist = decode_periods(temporal_schema, cube_start_date, cube_end_date, int(temporal_step))

        where = [Tile.grs_schema_id == self.datacube.grs_schema_id]

        if self.params.get('tiles'):
            where.append(Tile.id.in_(self.params['tiles']))

        self.tiles = Tile.query().filter(*where).all()

        self.bands = Band.query().filter(Band.collection_id == self.datacube.id).all()

        number_cols = self.datacube.raster_size_schemas.raster_size_x
        number_rows = self.datacube.raster_size_schemas.raster_size_y

        for tile in self.tiles:
            self.mosaics[tile.id] = dict(
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

                    self.mosaics[tile.id]['periods'][periodkey] = {}
                    self.mosaics[tile.id]['periods'][periodkey]['start'] = startdate
                    self.mosaics[tile.id]['periods'][periodkey]['end'] = enddate
                    self.mosaics[tile.id]['periods'][periodkey]['cols'] = number_cols
                    self.mosaics[tile.id]['periods'][periodkey]['rows'] = number_rows
                    self.mosaics[tile.id]['periods'][periodkey]['dirname']  = '{}/{}/{}-{}/'.format(self.datacube.id, tile.id, startdate, enddate)

    @property
    def warped_datacube(self):
        datacube_warped = self.datacube.id

        for fn in ['MEDIAN', 'STACK']:
            datacube_warped = datacube_warped.replace(fn, 'WARPED')

        return Collection.query().filter(Collection.id == datacube_warped).first()

    @property
    def datacube_bands(self):
        if self.params.get('bands'):
            return list(filter(lambda band: band.common_name in self.params['bands'], self.bands))
        return self.bands

    def prepare_merge(self):
        for tileid in self.mosaics:
            if len(self.mosaics) != 1:
                self.params['tileid'] = tileid
                continue

            bbox_result = db.session.query(
                Tile.id,
                func.ST_AsText(func.ST_BoundingDiagonal(func.ST_Force2D(Tile.geom_wgs84)))
            ).filter(
                Tile.id == tileid
            ).first()

            bbox = bbox_result[1][bbox_result[1].find('(') + 1:bbox_result[0].find(')')]
            bbox = bbox.replace(' ', ',')

            for periodkey in self.mosaics[tileid]['periods']:
                start = self.mosaics[tileid]['periods'][periodkey]['start']
                end = self.mosaics[tileid]['periods'][periodkey]['end']
                # activity['dirname'] = self.mosaics[tileid]['periods'][periodkey]['dirname']

                # Search all images
                self.mosaics[tileid]['periods'][periodkey]['scenes'] = self.search_images(bbox, start, end)

    @staticmethod
    def create_activity(collection: str, warped: str, activity_type: str, scene_type: str, band: str, period: str, **parameters):
        return dict(
            band=band,
            collection_id=collection,
            warped_collection_id=warped,
            activity_type=activity_type,
            tags=parameters.get('tags', []),
            period=period,
            scene_type=scene_type,
            args=parameters
        )

    def dispatch_celery(self):
        from celery import group, chain
        from .tasks import blend, warp_merge, publish
        self.prepare_merge()

        datacube = self.datacube.id

        if datacube is None:
            datacube = self.params['datacube']

        bands = self.datacube_bands
        warped_datacube = self.warped_datacube.id

        for tileid in self.mosaics:
            blends = []

            tile = next(filter(lambda t: t.id == tileid, self.tiles))

            # For each blend
            for period in self.mosaics[tileid]['periods']:
                merges_tasks = []

                cols = self.mosaics[tileid]['periods'][period]['cols']
                rows = self.mosaics[tileid]['periods'][period]['rows']
                start_date = self.mosaics[tileid]['periods'][period]['start']
                end_date = self.mosaics[tileid]['periods'][period]['end']
                period_start_end = '{}_{}'.format(start_date, end_date)

                for band in bands:
                    collections = self.mosaics[tileid]['periods'][period]['scenes'][band.common_name]

                    for collection, merges in collections.items():
                        for merge_date, assets in merges.items():
                            properties = dict(
                                date=merge_date,
                                dataset=collection,
                                xmin=tile.min_x,
                                ymax=tile.max_y,
                                datacube=datacube,
                                resx=band.resolution_x,
                                resy=band.resolution_y,
                            )

                            # activity = self.create_activity(
                            #     self.datacube.id,
                            #     self.warped_datacube.id,
                            #     'MERGE',
                            #     'WARPED',
                            #     band.id,
                            #     period_start_end,
                            #     **properties
                            # )

                            # Activity(**activity).save(commit=False)

                            # task = warp_merge.s(activity)
                            task = warp_merge.s(warped_datacube, tileid, period_start_end, assets, cols, rows, **properties)
                            merges_tasks.append(task)

                # Persist activities
                db.session.commit()

                task = chain(group(merges_tasks), blend.s())
                blends.append(task)

            task = group(blends)
            task.apply_async()

        return self.mosaics

    def search_images(self, bbox: str, start: str, end: str):
        scenes = {}
        options = dict(
            bbox=bbox,
            time='{}/{}'.format(start, end),
            limit=100000
        )

        bands = self.datacube_bands

        for band in bands:
            scenes[band.common_name] = dict()

        for dataset in self.params['collections']:
            collection_metadata = stac_cli.collection(dataset)

            collection_bands = collection_metadata['properties']['bdc:bands']

            items = stac_cli.collections[dataset].get_items(filter=options)

            for feature in items['features']:
                if feature['type'] == 'Feature':
                    date = feature['properties']['datetime'][0:10]
                    identifier = feature['id']
                    tile = feature['properties']['bdc:tile']

                    for band in bands:
                        if band.common_name not in feature['assets']:
                            continue

                        scenes[band.common_name].setdefault(dataset, dict())

                        link = feature['assets'][band.common_name]['href']

                        # radiometric_processing = linfeature['radiometric_processing']
                        # if radiometric_processing == 'DN' or radiometric_processing == 'TOA': continue

                        scene = {**collection_bands[band.common_name]}
                        scene['sceneid'] = identifier
                        scene['tile'] = tile
                        scene['date'] = date
                        scene['band'] = band.common_name
                        scene['link'] = link

                        if dataset == 'MOD13Q1' and band.common_name == 'quality':
                            scene['link'] = scene['link'].replace('quality','reliability')

                        scenes[band.common_name][dataset].setdefault(date, [])
                        scenes[band.common_name][dataset][date].append(scene)

        return scenes
