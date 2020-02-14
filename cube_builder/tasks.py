# Python Native
from copy import deepcopy
import logging
import traceback
# 3rdparty
from celery import chain, group
# BDC Scripts
from bdc_db.models import Collection

from cube_builder.models import Activity
from cube_builder.utils import get_or_create_activity
from .celery import celery_app
from .utils import merge as merge_processing, \
                   blend as blend_processing, \
                   publish_datacube, publish_merge


def capture_traceback(exception=None):
    return traceback.format_exc() or str(exception)


@celery_app.task()
def warp_merge(activity):
    logging.warning('Executing merge {}'.format(activity.get('warped_collection_id')))

    record: Activity = Activity.query().filter(Activity.id == activity['id']).one()

    # TODO: Validate in disk
    if record.status != 'SUCCESS':
        record.status = 'STARTED'
        record.save()

        try:
            args = deepcopy(activity.get('args'))
            period = activity.get('period')
            warped_datacube = activity.get('warped_collection_id')
            tile_id = args.pop('tile_id')
            assets = args.pop('assets')
            cols = args.pop('cols')
            rows = args.pop('rows')

            res = merge_processing(warped_datacube, tile_id, assets, int(cols), int(rows), period, **args)

            merge_args = activity['args']
            merge_args.update(res)

            record.traceback = ''
            record.status = 'SUCCESS'
            record.args = merge_args
        except BaseException as e:
            record.status = 'FAILURE'
            record.traceback = capture_traceback(e)
            logging.error('Error in merge. Activity {}'.format(record.id), exc_info=True)

            raise e
        finally:
            record.save()

    return activity


@celery_app.task()
def blend(merges):
    activities = dict()

    for _merge in merges:
        if _merge['band'] in activities and _merge['args']['date'] in activities[_merge['band']]['scenes'] or \
                _merge['band'] == 'quality':
            continue

        activity = activities.get(_merge['band'], dict(scenes=dict()))

        activity['datacube'] = _merge['collection_id']
        activity['warped_datacube'] = merges[0]['warped_collection_id']
        activity['band'] = _merge['band']
        activity['scenes'].setdefault(_merge['args']['date'], dict(**_merge['args']))
        activity['period'] = _merge['period']
        activity['tile_id'] = _merge['args']['tile_id']

        activity['scenes'][_merge['args']['date']]['ARDfiles'] = {
            "quality": _merge['args']['file'].replace(_merge['band'], 'quality'),
            _merge['band']: _merge['args']['file']
        }

        activities[_merge['band']] = activity

    logging.warning('Scheduling blend....')

    blends = []

    for activity in activities.values():
        # TODO: Persist
        blends.append(_blend.s(activity))

    task = chain(group(blends), publish.s())
    task.apply_async()


@celery_app.task()
def _blend(activity):
    logging.warning('Executing blend - {} - {}'.format(activity.get('datacube'), activity.get('band')))

    return blend_processing(activity)


@celery_app.task()
def publish(blends):
    logging.warning('Executing publish')

    cube = Collection.query().filter(Collection.id == blends[0]['datacube']).first()
    warped_datacube = blends[0]['warped_datacube']
    tile_id = blends[0]['tile_id']
    period = blends[0]['period']
    cloudratio = blends[0]['cloudratio']

    # Retrieve which bands to generate quick look
    quick_look_bands = cube.bands_quicklook.split(',')

    merges = dict()
    blend_files = dict()

    for blend_result in blends:
        blend_files[blend_result['band']] = blend_result['blends']

        for merge_date, definition in blend_result['scenes'].items():
            merges.setdefault(merge_date, dict(dataset=definition['dataset'], ARDfiles=dict()))
            merges[merge_date]['ARDfiles'].update(definition['ARDfiles'])

    # Generate quick looks for cube scenes
    publish_datacube(cube, quick_look_bands, cube.id, tile_id, period, blend_files, cloudratio)

    # Generate quick looks of irregular cube
    # In order to do that, we must schedule new celery tasks and execute in parallel
    tasks = []

    for merge_date, definition in merges.items():
        date = merge_date.replace(definition['dataset'], '')
        tasks.append(_publish_merge.s(quick_look_bands, warped_datacube, definition['dataset'], tile_id, period, date, definition))

    promise = chain(group(tasks))
    promise.apply_async()


@celery_app.task()
def _publish_merge(bands, datacube, dateset, tile_id, period, merge_date, scenes):
    return publish_merge(bands, datacube, dateset, tile_id, period, merge_date, scenes)
