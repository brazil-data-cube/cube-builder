#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# Python Native
from os import path as resource_path
import logging
# 3rdparty
from celery import chain, group
# BDC Scripts
from bdc_db.models import Collection
from .celery import celery_app
from .utils import merge as merge_processing, \
                   blend as blend_processing, \
                   publish_datacube, publish_merge


@celery_app.task()
def warp_merge(warped_datacube, tile_id, period, warps, cols, rows, **kwargs):
    logging.warning('Executing merge {}'.format(kwargs.get('datacube')), kwargs)

    return merge_processing(warped_datacube, tile_id, warps, int(cols), int(rows), period, **kwargs)


@celery_app.task()
def blend(merges):
    activities = dict()

    for _merge in merges:
        if _merge['band'] in activities and _merge['date'] in activities[_merge['band']]['scenes']:
            continue

        activity = activities.get(_merge['band'], dict(scenes=dict()))

        activity['datacube'] = _merge['datacube']
        activity['warped_datacube'] = merges[0]['warped_datacube']
        activity['band'] = _merge['band']
        activity['scenes'].setdefault(_merge['date'], dict(**_merge))
        activity['period'] = _merge['period']
        activity['tile_id'] = _merge['tile_id']

        activity['scenes'][_merge['date']]['ARDfiles'] = {
            "quality": _merge['file'].replace(_merge['band'], 'quality'),
            _merge['band']: _merge['file']
        }

        activities[_merge['band']] = activity

    logging.warning('Scheduling blend....')

    blends = []

    for activity in activities.values():
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
    publish_datacube(quick_look_bands, cube.id, tile_id, period, blend_files)

    # Generate quick looks of irregular cube
    # In order to do that, we must schedule new celery tasks and execute in parallel
    tasks = []

    for merge_date, definition in merges.items():
        date = merge_date.replace(definition['dataset'], '')
        tasks.append(_publish_merge.s(quick_look_bands, warped_datacube, definition['dataset'], tile_id, period, date, definition))

    promise = chain(group(tasks), upload.s())
    promise.apply_async()


@celery_app.task()
def _publish_merge(bands, datacube, dateset, tile_id, period, merge_date, scenes):
    return publish_merge(bands, datacube, dateset, tile_id, period, merge_date, scenes)


@celery_app.task()
def upload(*args, **kwargs):
    pass
