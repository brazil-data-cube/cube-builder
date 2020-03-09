#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define celery tasks for Cube Builder."""

# Python Native
import logging
import traceback
from copy import deepcopy

# 3rdparty
from bdc_db.models import Collection
from celery import chain, group

# Cube Builder
from .celery import celery_app
from .models import Activity
from .utils import blend as blend_processing
from .utils import merge as merge_processing
from .utils import publish_datacube, publish_merge


def capture_traceback(exception=None):
    """Retrieve stacktrace as string."""
    return traceback.format_exc() or str(exception)


@celery_app.task()
def warp_merge(activity, force=False):
    """Execute datacube merge task.

    This task consists in the following steps:

    **1.** Prepare a raster using dimensions of datacube GRS schema.
    **2.** Open collection dataset with RasterIO and reproject to datacube GRS Schema.
    **3.** Fill the respective pathrow into raster

    Args:
        activity - Datacube Activity Model
        force - Flag to build datacube without cache.

    Returns:
        Validated activity
    """
    logging.warning('Executing merge {}'.format(activity.get('warped_collection_id')))

    record: Activity = Activity.query().filter(Activity.id == activity['id']).one()

    # TODO: Validate in disk
    if force or record.status != 'SUCCESS':
        record.status = 'STARTED'
        record.save()

        try:
            args = deepcopy(activity.get('args'))
            warped_datacube = activity.get('warped_collection_id')
            _ = args.pop('period', None)
            merge_date = args.get('date')

            # res = merge_processing(warped_datacube, tile_id, assets, int(cols), int(rows), merge_date, **args)
            res = merge_processing(warped_datacube, period=merge_date, **args)

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
    """Receive merges and prepare task blend.

    This task aims to prepare celery task definition for blend.
    A blend requires both data set quality band and others bands. In this way, we must group
    these values by temporal resolution and then schedule blend tasks.
    """
    activities = dict()

    for _merge in merges:
        if _merge['band'] in activities and _merge['args']['date'] in activities[_merge['band']]['scenes'] or \
                _merge['band'] == 'quality':
            continue

        activity = activities.get(_merge['band'], dict(scenes=dict()))

        activity['datacube'] = _merge['collection_id']
        activity['warped_datacube'] = _merge['warped_collection_id']
        activity['band'] = _merge['band']
        activity['scenes'].setdefault(_merge['args']['date'], dict(**_merge['args']))
        activity['period'] = _merge['period']
        activity['tile_id'] = _merge['args']['tile_id']
        activity['nodata'] = _merge['args'].get('nodata')

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
    """Execute datacube blend task.

    TODO: Describe how it works.

    Args:
        activity - Datacube Activity Model

    Returns:
        Validated activity
    """
    logging.warning('Executing blend - {} - {}'.format(activity.get('datacube'), activity.get('band')))

    return blend_processing(activity)


@celery_app.task()
def publish(blends):
    """Execute publish task and catalog datacube result.

    Args:
        activity - Datacube Activity Model
    """
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
            merges.setdefault(merge_date, dict(dataset=definition['dataset'],
                                               cloudratio=definition['cloudratio'],
                                               ARDfiles=dict()))
            merges[merge_date]['ARDfiles'].update(definition['ARDfiles'])

    # Generate quick looks for cube scenes
    publish_datacube(cube, quick_look_bands, cube.id, tile_id, period, blend_files, cloudratio)

    # Generate quick looks of irregular cube
    for merge_date, definition in merges.items():
        date = merge_date.replace(definition['dataset'], '')

        wcube = Collection.query().filter(Collection.id == warped_datacube).first()

        publish_merge(quick_look_bands, wcube, definition['dataset'], tile_id, period, date, definition)
