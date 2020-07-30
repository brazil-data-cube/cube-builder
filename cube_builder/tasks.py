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

# 3rdparty
from bdc_db.models import AssetMV, Collection, db
from celery import chain, group
from sqlalchemy_utils import refresh_materialized_view

# Cube Builder
from .celery import celery_app
from .constants import CLEAR_OBSERVATION_NAME, TOTAL_OBSERVATION_NAME, PROVENANCE_NAME
from .models import Activity
from .utils import DataCubeFragments, blend as blend_processing, build_cube_path, post_processing_quality, get_cube_id
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
        period=activity.get('period'),
        date=activity.get('date'),
        tile_id=activity.get('tile_id')
    )

    model, created = get_or_create_model(Activity, defaults=activity, **where)

    logging.debug('Activity {}, {}, {}, {} - {}'.format(model.tile_id, model.band, model.date,
                                                        model.collection_id, created))

    return model


@celery_app.task()
def warp_merge(activity, band_map, force=False):
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

    merge_file_path = build_cube_path(record.warped_collection_id, record.band,
                                      merge_date.replace(data_set, ''), tile_id)

    # Reuse merges already done. Rebuild only with flag ``--force``
    if not force and merge_file_path.exists() and merge_file_path.is_file():
        efficacy = cloudratio = 0

        if activity['band'] == band_map['quality']:
            # When file exists, compute the file statistics
            efficacy, cloudratio = compute_data_set_stats(str(merge_file_path))

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
            args.pop('period', None)
            args['tile_id'] = tile_id
            args['date'] = record.date.strftime('%Y-%m-%d')
            args['cube'] = record.warped_collection_id

            res = merge_processing(str(merge_file_path), band_map=band_map, **args)

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
def prepare_blend(merges, band_map: dict, **kwargs):
    """Receive merges by period and prepare task blend.

    This task aims to prepare celery task definition for blend.
    A blend requires both data set quality band and others bands. In this way, we must group
    these values by temporal resolution and then schedule blend tasks.
    """
    activities = dict()

    # Prepare map of efficacy/cloud_ratio based in quality merge result
    quality_date_stats = {
        m['date']: (m['args']['efficacy'], m['args']['cloudratio'], m['args']['file']) for m in merges if m['band'] == band_map['quality']
    }

    for period, stats in quality_date_stats.items():
        _, _, quality_file = stats

        logging.info(f'Applying post-processing in {str(quality_file)}')
        post_processing_quality(quality_file, list(band_map.values()), merges[0]['warped_collection_id'],
                                period, merges[0]['tile_id'], band_map['quality'])

    def _is_not_stk(_merge):
        """Control flag to generate cloud mask.

        This function is a utility to dispatch the cloud mask generation only for STK data cubes.
        """
        return _merge['band'] == band_map['quality'] and not _merge['collection_id'].endswith('STK')

    composite_functions = kwargs.get('composite_functions', [])

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
        activity['composite_functions'] = composite_functions

        # Map efficacy/cloud ratio to the respective merge date before pass to blend
        efficacy, cloudratio, quality_file = quality_date_stats[_merge['date']]
        activity['scenes'][_merge['args']['date']]['efficacy'] = efficacy
        activity['scenes'][_merge['args']['date']]['cloudratio'] = cloudratio

        activity['scenes'][_merge['args']['date']]['ARDfiles'] = {
            band_map['quality']: quality_file,
            _merge['band']: _merge['args']['file']
        }

        activities[_merge['band']] = activity

    # Prepare list of activities to dispatch
    activity_list = list(activities.values())

    datacube = activity_list[0]['datacube']

    # For IDENTITY data cube trigger, just publish
    if DataCubeFragments(datacube).composite_function == 'IDENTITY':
        task = publish.s(list(activities.values()))
        return task.apply_async()

    logging.warning('Scheduling blend....')

    blends = []

    # We must keep track of last activity to run
    # Since the Clear Observation must only be execute by single process. It is important
    # to avoid concurrent processes to write same data set in disk
    last_activity = activity_list[-1]

    # Trigger all except the last
    for activity in activity_list[:-1]:
        # TODO: Persist
        blends.append(blend.s(activity, band_map))

    # Trigger last blend to execute Clear Observation
    blends.append(blend.s(last_activity, band_map, build_clear_observation=True))

    task = chain(group(blends), publish.s(band_map, **kwargs))
    task.apply_async()


@celery_app.task()
def blend(activity, band_map, build_clear_observation=False):
    """Execute datacube blend task.

    Args:
        activity - Datacube Activity Model.
        band_map - Band mapping with common_name and band original name.
        build_clear_observation - Generate band "Clear Observation".

    Returns:
        Validated activity
    """
    logging.warning('Executing blend - {} - {}'.format(activity.get('datacube'), activity.get('band')))

    return blend_processing(activity, band_map, build_clear_observation)


@celery_app.task()
def publish(blends, band_map, **kwargs):
    """Execute publish task and catalog datacube result.

    Args:
        activity - Datacube Activity Model
    """
    period = blends[0]['period']
    logging.info(f'Executing publish {period}')

    cube = Collection.query().filter(Collection.id == blends[0]['datacube']).first()
    warped_datacube = blends[0]['warped_datacube']
    tile_id = blends[0]['tile_id']

    # Retrieve which bands to generate quick look
    quick_look_bands = cube.bands_quicklook.split(',')

    merges = dict()
    blend_files = dict()

    composite_function = DataCubeFragments(cube.id).composite_function

    for blend_result in blends:
        if composite_function != 'IDENTITY':
            blend_files[blend_result['band']] = blend_result['blends']

        if blend_result.get('clear_observation_file'):
            blend_files[CLEAR_OBSERVATION_NAME] = blend_result['clear_observation_file']

        if blend_result.get('total_observation'):
            blend_files[TOTAL_OBSERVATION_NAME] = blend_result['total_observation']

        if blend_result.get('provenance'):
            blend_files[PROVENANCE_NAME] = {composite_function: blend_result['provenance']}

        for merge_date, definition in blend_result['scenes'].items():
            merges.setdefault(merge_date, dict(dataset=definition['dataset'],
                                               cloudratio=definition['cloudratio'],
                                               ARDfiles=dict()))
            merges[merge_date]['ARDfiles'].update(definition['ARDfiles'])

    if composite_function != 'IDENTITY':
        cloudratio = blends[0]['cloudratio']

        # Generate quick looks for cube scenes
        publish_datacube(cube, quick_look_bands, cube.id, tile_id, period, blend_files, cloudratio, band_map, **kwargs)

    # Generate quick looks of irregular cube
    wcube = Collection.query().filter(Collection.id == warped_datacube).first()

    for merge_date, definition in merges.items():
        date = merge_date.replace(definition['dataset'], '')

        publish_merge(quick_look_bands, wcube, tile_id, period, date, definition, band_map)

    try:
        refresh_materialized_view(db.session, AssetMV.__table__)
        db.session.commit()
        logging.info('View refreshed.')
    except:
        db.session.rollback()
