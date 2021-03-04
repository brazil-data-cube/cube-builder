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
from bdc_catalog.models import Collection, db
from celery import chain, group

# Cube Builder
from ..celery import celery_app
from ..models import Activity
from ..constants import CLEAR_OBSERVATION_NAME, TOTAL_OBSERVATION_NAME, PROVENANCE_NAME, DATASOURCE_NAME
from ..utils.image import create_empty_raster, match_histogram_with_merges
from ..utils.processing import DataCubeFragments, build_cube_path, post_processing_quality
from ..utils.processing import compute_data_set_stats, get_or_create_model
from ..utils.processing import blend as blend_processing, merge as merge_processing, publish_datacube, publish_merge


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


@celery_app.task(queue='merge-cube')
def warp_merge(activity, band_map, force=False, **kwargs):
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
    version = activity['args']['version']

    merge_file_path = None

    if activity['args'].get('reuse_datacube'):
        collection = Collection.query().filter(Collection.id == activity['args']['reuse_datacube']).first()

        if not force:
            # TODO: Should we search in Activity instead?
            merge_file_path = build_cube_path(collection.name, merge_date, tile_id,
                                              version=collection.version, band=record.band)

            if not merge_file_path.exists():
                # TODO: Should we raise exception??
                logging.warning(f'Cube {record.warped_collection_id} requires {collection.name}, but the file {str(merge_file_path)} not found. Skipping')
                raise RuntimeError(
                    f"""Cube {record.warped_collection_id} is derived from {collection.name},
                    but the file {str(merge_file_path)} was not found."""
                )

        else:
            raise RuntimeError(
                f'Cannot use option "force" for derived data cube - {record.warped_collection_id} of {collection.name}'
            )

    if merge_file_path is None:
        merge_file_path = build_cube_path(record.warped_collection_id, merge_date,
                                          tile_id, version=version, band=record.band)

        if activity['band'] == band_map['quality'] and len(activity['args']['datasets']):
            kwargs['build_provenance'] = True

    reused = False

    # Reuse merges already done. Rebuild only with flag ``--force``
    if not force and merge_file_path.exists() and merge_file_path.is_file():
        efficacy = cloudratio = 0

        if activity['band'] == band_map['quality']:
            # When file exists, compute the file statistics
            efficacy, cloudratio = compute_data_set_stats(str(merge_file_path))

        reused = True

        activity['args']['file'] = str(merge_file_path)
        activity['args']['efficacy'] = efficacy
        activity['args']['cloudratio'] = cloudratio
        record.traceback = ''

        args = deepcopy(record.args)
        args.update(activity['args'])

        activity['args'] = args

        record.args = args
        record.save()
    else:
        record.status = 'STARTED'
        record.save()

        record.args = activity['args']

        try:
            args = deepcopy(activity.get('args'))
            args.pop('period', None)
            args['tile_id'] = tile_id
            args['date'] = record.date.strftime('%Y-%m-%d')
            args['cube'] = record.warped_collection_id

            empty = args.get('empty', False)

            # Create base directory
            merge_file_path.parent.mkdir(parents=True, exist_ok=True)

            if empty:
                # create empty raster
                file_path = create_empty_raster(str(merge_file_path),
                                                proj4=args['srs'],
                                                cog=True,
                                                nodata=args['nodata'],
                                                dtype='int16',  # TODO: Pass through args
                                                dist=[args['dist_x'], args['dist_y']],
                                                resolution=[args['resx'], args['resy']],
                                                xmin=args['xmin'],
                                                ymax=args['ymax'])
                res = dict(
                    file=str(file_path),
                    efficacy=100,
                    cloudratio=0,
                    resolution=args['resx'],
                    nodata=args['nodata']
                )
            else:
                res = merge_processing(str(merge_file_path), band_map=band_map, band=record.band, **args, **kwargs)

            merge_args = deepcopy(activity['args'])
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
        activity['args']['efficacy'],
        activity['args']['cloudratio']
    ))

    activity['args']['reused'] = reused

    return activity


@celery_app.task(queue='prepare-cube')
def prepare_blend(merges, band_map: dict, **kwargs):
    """Receive merges by period and prepare task blend.

    This task aims to prepare celery task definition for blend.
    A blend requires both data set quality band and others bands. In this way, we must group
    these values by temporal resolution and then schedule blend tasks.
    """
    block_size = kwargs.get('block_size')

    activities = dict()

    # Prepare map of efficacy/cloud_ratio based in quality merge result
    quality_date_stats = {
        m['date']: (m['args']['efficacy'], m['args']['cloudratio'], m['args']['file'], m['args']['reused'])
        for m in merges if m['band'] == band_map['quality']
    }

    version = merges[0]['args']['version']

    for period, stats in quality_date_stats.items():
        _, _, quality_file, was_reused = stats

        # Do not apply post-processing on reused data cube since it may be already processed.
        if not was_reused:
            logging.info(f'Applying post-processing in {str(quality_file)}')
            post_processing_quality(quality_file, list(band_map.values()), merges[0]['warped_collection_id'],
                                    period, merges[0]['tile_id'], band_map['quality'], version=version, block_size=block_size)
        else:
            logging.info(f'Skipping post-processing {str(quality_file)}')

    def _is_not_stk(_merge):
        """Control flag to generate cloud mask.

        This function is a utility to dispatch the cloud mask generation only for STK data cubes.
        """
        return _merge['band'] == band_map['quality'] and not _merge['collection_id'].endswith('STK')

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
        activity['version'] = version
        # TODO: Check instance type for backward compatibility
        activity['datasets'] = _merge['args']['datasets']

        # Map efficacy/cloud ratio to the respective merge date before pass to blend
        efficacy, cloudratio, quality_file, _ = quality_date_stats[_merge['date']]
        activity['scenes'][_merge['args']['date']]['efficacy'] = efficacy
        activity['scenes'][_merge['args']['date']]['cloudratio'] = cloudratio

        if _merge['args'].get('reuse_datacube'):
            activity['reuse_datacube'] = _merge['args']['reuse_datacube']

        activity['scenes'][_merge['args']['date']]['ARDfiles'] = {
            band_map['quality']: quality_file,
            _merge['band']: _merge['args']['file']
        }

        if _merge['args'].get(DATASOURCE_NAME):
            activity['scenes'][_merge['args']['date']]['ARDfiles'][DATASOURCE_NAME] = _merge['args'][DATASOURCE_NAME]

        activities[_merge['band']] = activity

    # TODO: Add option to skip histogram.
    if kwargs.get('histogram_matching'):
        ordered_best_efficacy = sorted(quality_date_stats.items(), key=lambda item: item[1][0], reverse=True)

        best_date, (_, _, best_mask_file, _) = ordered_best_efficacy[0]
        dates = map(lambda entry: entry[0], ordered_best_efficacy[1:])

        for date in dates:
            logging.info(f'Applying Histogram Matching: Reference date {best_date}, current {date}...')
            for band, activity in activities.items():
                reference = activities[band]['scenes'][best_date]['ARDfiles'][band]

                if band == band_map['quality']:
                    continue

                source = activity['scenes'][date]['ARDfiles'][band]
                source_mask = activity['scenes'][date]['ARDfiles'][band_map['quality']]
                match_histogram_with_merges(source, source_mask, reference, best_mask_file, block_size=block_size)

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


@celery_app.task(queue='blend-cube')
def blend(activity, band_map, build_clear_observation=False, **kwargs):
    """Execute datacube blend task.

    Args:
        activity - Datacube Activity Model.
        band_map - Band mapping with common_name and band original name.
        build_clear_observation - Generate band "Clear Observation".

    Returns:
        Validated activity
    """
    block_size = kwargs.get('block_size')

    logging.warning('Executing blend - {} - {}'.format(activity.get('datacube'), activity.get('band')))

    return blend_processing(activity, band_map, build_clear_observation, block_size=block_size)


@celery_app.task(queue='publish-cube')
def publish(blends, band_map, **kwargs):
    """Execute publish task and catalog datacube result.

    Args:
        activity - Datacube Activity Model
    """
    period = blends[0]['period']
    logging.info(f'Executing publish {period}')

    version = blends[0]['version']

    cube: Collection = Collection.query().filter(
        Collection.name == blends[0]['datacube'],
        Collection.version == version
    ).first()
    warped_datacube = blends[0]['warped_datacube']
    tile_id = blends[0]['tile_id']
    reused_cube = blends[0].get('reuse_datacube')

    # Retrieve which bands to generate quick look
    bands = cube.bands
    band_id_map = {band.id: band.name for band in bands}

    quicklook = cube.quicklook[0]

    quick_look_bands = [band_id_map[quicklook.red], band_id_map[quicklook.green], band_id_map[quicklook.blue]]

    merges = dict()
    blend_files = dict()

    composite_function = DataCubeFragments(cube.name).composite_function

    quality_blend = None

    for blend_result in blends:
        if composite_function != 'IDENTITY':
            blend_files[blend_result['band']] = blend_result['blends']

        if blend_result.get('clear_observation_file'):
            blend_files[CLEAR_OBSERVATION_NAME] = {composite_function: blend_result['clear_observation_file']}

        if blend_result.get('total_observation'):
            blend_files[TOTAL_OBSERVATION_NAME] = {composite_function: blend_result['total_observation']}

        if blend_result.get('provenance'):
            blend_files[PROVENANCE_NAME] = {composite_function: blend_result['provenance']}

        if blend_result.get('datasource'):
            blend_files[DATASOURCE_NAME] = {composite_function: blend_result['datasource']}

        for merge_date, definition in blend_result['scenes'].items():
            merges.setdefault(merge_date, dict(datasets=definition.get('datasets', definition.get('dataset')),
                                               cloudratio=definition['cloudratio'],
                                               ARDfiles=dict()))
            merges[merge_date]['ARDfiles'].update(definition['ARDfiles'])

        if blend_result['band'] == band_map['quality']:
            quality_blend = blend_result

    if composite_function != 'IDT':
        cloudratio = quality_blend['cloudratio']

        # Generate quick looks for cube scenes
        publish_datacube(cube, quick_look_bands, tile_id, period, blend_files, cloudratio, band_map, **kwargs)

    # Generate quick looks of irregular cube
    wcube = Collection.query().filter(Collection.name == warped_datacube, Collection.version == version).first()

    if not reused_cube:
        for merge_date, definition in merges.items():
            publish_merge(quick_look_bands, wcube, tile_id, merge_date, definition, band_map)

        try:
            db.session.commit()
        except:
            db.session.rollback()
