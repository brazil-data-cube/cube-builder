#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define celery tasks for Cube Builder."""

# Python Native
import logging
import traceback
from copy import deepcopy
from pathlib import Path

# 3rdparty
from bdc_db.models import AssetMV, Collection, db
from celery import chain, group
from sqlalchemy_utils import refresh_materialized_view

# Cube Builder
from .celery import celery_app
from .config import Config
from .models import Activity
from .utils import blend as blend_processing
from .utils import compute_data_set_stats, get_or_create_model
from .utils import merge as merge_processing
from .utils import publish_datacube, publish_merge


def capture_traceback(exception=None):
    """Retrieve stacktrace as string."""
    return traceback.format_exc() or str(exception)


def create_execution(activity: dict) -> Activity:
    """Create cube-builder activity and prepare celery execution.

    Args:
        activity - Cube Builder Activity dict

    Returns:
        Activity the cube build activity model
    """
    where = dict(
        band=activity.get('band'),
        collection_id=activity.get('collection_id'),
        period=activity.get('period'),
        date=activity.get('date'),
        tile_id=activity.get('tile_id')
    )

    model, created = get_or_create_model(Activity, defaults=activity, **where)

    logging.debug('Activity {}, {}, {}, {} - {}'.format(model.tile_id, model.band, model.date,
                                                        model.collection_id, created))

    return model


@celery_app.task()
def warp_merge(activity, force=False):
    """Execute datacube merge task.

    This task consists in the following steps:

    **1.** Prepare a raster using dimensions of datacube GRS schema.
    **2.** Open collection dataset with RasterIO and reproject to datacube GRS Schema.
    **3.** Fill the respective pathrow into raster

    Args:
        activity - Datacube Activity Model
        force - Flag to build data cube without cache.

    Returns:
        Validated activity
    """
    logging.warning('Executing merge {} - {}'.format(activity.get('warped_collection_id'), activity['band']))

    record = create_execution(activity)

    record.warped_collection_id = activity['warped_collection_id']
    merge_date = activity['date']
    tile_id = activity['tile_id']
    data_set = activity['args'].get('dataset')

    merge_name = '{}_{}_{}_{}'.format(record.warped_collection_id, tile_id, merge_date, record.band)

    merge_file_path = (Path(Config.DATA_DIR) / 'Repository/Warped') / '{}/{}/{}/{}.tif'.format(
        record.warped_collection_id,
        tile_id,
        merge_date.replace(data_set, ''),
        merge_name
    )

    # Reuse merges already done. Rebuild only with flag ``--force``
    if not force and merge_file_path.exists() and merge_file_path.is_file():
        efficacy = cloudratio = 0

        if activity['band'] == 'quality':
            # When file exists, compute the file statistics
            efficacy, cloudratio = compute_data_set_stats(merge_file_path)

        activity['args']['file'] = str(merge_file_path)
        activity['args']['efficacy'] = efficacy
        activity['args']['cloudratio'] = cloudratio
        record.traceback = ''
        record.args = activity['args']
        record.save()
    else:
        record.status = 'STARTED'
        record.args = activity['args']
        record.save()

        try:
            args = deepcopy(activity.get('args'))
            _ = args.pop('period', None)

            res = merge_processing(merge_file_path, **args)

            merge_args = activity['args']
            merge_args.update(res)

            record.traceback = ''
            record.status = 'SUCCESS'
            record.args = merge_args

            activity['args'].update(merge_args)
        except BaseException as e:
            record.status = 'FAILURE'
            record.traceback = capture_traceback(e)
            logging.error('Error in merge. Activity {}'.format(record.id), exc_info=True)

            raise e
        finally:
            record.save()

    logging.warning('Merge {} executed successfully. Efficacy={}, cloud_ratio={}'.format(
        str(merge_file_path),
        record.args['efficacy'],
        record.args['cloudratio']
    ))

    return activity


@celery_app.task()
def prepare_blend(merges):
    """Receive merges by period and prepare task blend.

    This task aims to prepare celery task definition for blend.
    A blend requires both data set quality band and others bands. In this way, we must group
    these values by temporal resolution and then schedule blend tasks.
    """
    activities = dict()

    # Prepare map of efficacy/cloud_ratio based in quality merge result
    quality_date_stats = {
        m['date']: (m['args']['efficacy'], m['args']['cloudratio']) for m in merges if m['band'] == 'quality'
    }

    def _is_not_stk(_merge):
        """Control flag to generate cloud mask.

        This function is a utility to dispatch the cloud mask generation only for STK data cubes.
        """
        return _merge['band'] == 'quality' and not _merge['collection_id'].endswith('STK')

    for _merge in merges:
        # Skip quality generation for MEDIAN, AVG
        if _merge['band'] in activities and _merge['args']['date'] in activities[_merge['band']]['scenes'] or \
                _is_not_stk(_merge):
            continue

        activity = activities.get(_merge['band'], dict(scenes=dict()))

        activity['datacube'] = _merge['collection_id']
        activity['warped_datacube'] = _merge['warped_collection_id']
        activity['band'] = _merge['band']
        activity['scenes'].setdefault(_merge['args']['date'], dict(**_merge['args']))
        activity['period'] = _merge['period']
        activity['tile_id'] = _merge['tile_id']
        activity['nodata'] = _merge['args'].get('nodata')

        # Map efficacy/cloud ratio to the respective merge date before pass to blend
        efficacy, cloudratio = quality_date_stats[_merge['date']]
        activity['scenes'][_merge['args']['date']]['efficacy'] = efficacy
        activity['scenes'][_merge['args']['date']]['cloudratio'] = cloudratio

        activity['scenes'][_merge['args']['date']]['ARDfiles'] = {
            "quality": _merge['args']['file'].replace(_merge['band'], 'quality'),
            _merge['band']: _merge['args']['file']
        }

        activities[_merge['band']] = activity

    # TODO: Generate Vegetation Index

    logging.warning('Scheduling blend....')

    blends = []

    # Prepare list of activities to dispatch
    activity_list = list(activities.values())

    # We must keep track of last activity to run
    # Since the Count No Cloud (CNC) must only be execute by single process. It is important
    # to avoid concurrent processes to write same data set in disk
    last_activity = activity_list[-1]

    # Trigger all except the last
    for activity in activity_list[:-1]:
        # TODO: Persist
        blends.append(blend.s(activity))

    # Trigger last blend to execute CNC
    blends.append(blend.s(last_activity, build_cnc=True))

    task = chain(group(blends), publish.s())
    task.apply_async()


@celery_app.task()
def blend(activity, build_cnc=False):
    """Execute datacube blend task.

    TODO: Describe how it works.

    Args:
        activity - Datacube Activity Model

    Returns:
        Validated activity
    """
    logging.warning('Executing blend - {} - {}'.format(activity.get('datacube'), activity.get('band')))

    return blend_processing(activity, build_cnc)


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

        if blend_result.get('cloud_count_file'):
            blend_files['cnc'] = dict(MED=blend_result['cloud_count_file'], STK=blend_result['cloud_count_file'])

        for merge_date, definition in blend_result['scenes'].items():
            merges.setdefault(merge_date, dict(dataset=definition['dataset'],
                                               cloudratio=definition['cloudratio'],
                                               ARDfiles=dict()))
            merges[merge_date]['ARDfiles'].update(definition['ARDfiles'])

    # Generate quick looks for cube scenes
    publish_datacube(cube, quick_look_bands, cube.id, tile_id, period, blend_files, cloudratio)

    # Generate quick looks of irregular cube
    wcube = Collection.query().filter(Collection.id == warped_datacube).first()

    for merge_date, definition in merges.items():
        date = merge_date.replace(definition['dataset'], '')

        publish_merge(quick_look_bands, wcube, definition['dataset'], tile_id, period, date, definition)

    try:
        refresh_materialized_view(db.session, AssetMV.__table__)
        db.session.commit()
        logging.info('View refreshed.')
    except:
        db.session.rollback()