#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define celery tasks for Cube Builder."""

# Python Native
import logging
import traceback
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from bdc_catalog.models import Collection, db
from celery import chain, group

# Cube Builder
from ..config import Config
from ..constants import CLEAR_OBSERVATION_NAME, DATASOURCE_NAME, PROVENANCE_NAME, TOTAL_OBSERVATION_NAME
from ..models import Activity
from ..utils import get_srid_column
from ..utils.image import check_file_integrity, create_empty_raster, match_histogram_with_merges
from ..utils.processing import DataCubeFragments
from ..utils.processing import blend as blend_processing
from ..utils.processing import build_cube_path, compute_data_set_stats, get_cube_id, get_item_id, get_or_create_model
from ..utils.processing import merge as merge_processing
from ..utils.processing import post_processing_quality, publish_datacube, publish_merge
from ..utils.timeline import temporal_priority_timeline
from . import celery_app
from .utils import clear_merge


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


@celery_app.task(queue=Config.QUEUE_IDENTITY_CUBE)
def warp_merge(activity, band_map, mask, force=False, data_dir=None, **kwargs):
    """Execute datacube merge task.

    This task consists in the following steps:

        - Prepare a raster using dimensions of datacube GRS schema.
        - Open collection dataset with RasterIO and reproject to datacube GRS Schema.
        - Fill the respective path row into raster

    Args:
        activity - Datacube Activity Model
        force - Flag to build data cube without cache.

    Returns:
        Validated activity
    """
    logging.warning('Executing merge {} - {}'.format(activity.get('warped_collection_id'), activity['band']))

    record = create_execution(activity)

    data_dir = data_dir or Config.DATA_DIR

    record.warped_collection_id = activity['warped_collection_id']
    merge_date = activity['date']

    tile_id = activity['tile_id']
    version = activity['args']['version']

    merge_file_path = None
    quality_band = kwargs.get('quality_band')
    collection_id = f'{record.collection_id}-{version}'

    if kwargs.get('reuse_data_cube'):
        ref_cube_idt = get_cube_id(kwargs['reuse_data_cube']['name'])
        # TODO: Should we search in Activity instead?
        merge_file_path = build_cube_path(ref_cube_idt, merge_date, tile_id,
                                          version=kwargs['reuse_data_cube']['version'], band=record.band,
                                          prefix=data_dir)  # check published dir

    if merge_file_path is None:
        merge_file_path = build_cube_path(record.warped_collection_id, merge_date,
                                          tile_id, version=version, band=record.band,
                                          prefix=data_dir)

        if activity['band'] == quality_band and len(activity['args']['datasets']):
            kwargs['build_provenance'] = True

    reused = False

    is_valid_or_exists = not force and merge_file_path.exists() and merge_file_path.is_file()

    # When is false, we must change to Work_dir context
    if not is_valid_or_exists:
        merge_file_path = Path(Config.WORK_DIR) / merge_file_path.relative_to(data_dir)
        is_valid_or_exists = not force and merge_file_path.exists() and merge_file_path.is_file()

    # Reuse merges already done. Rebuild only with flag ``--force``
    if is_valid_or_exists:
        efficacy = cloudratio = 0

        try:
            if not check_file_integrity(merge_file_path):
                raise IOError('Invalid Merge File')

            if activity['band'] == quality_band:
                # When file exists, compute the file statistics
                efficacy, cloudratio = compute_data_set_stats(str(merge_file_path), mask=mask, compute=True)
                datasets = activity['args']['datasets']
                if len(datasets) > 1:
                    # prepare datasource. TODO: Check with reuse data cube
                    item_id = get_item_id(record.warped_collection_id, version, tile_id, merge_date)
                    datasource_file = merge_file_path.parent / f'{item_id}_DATASOURCE.tif'
                    if not datasource_file.exists():
                        # remove cloud mask file
                        merge_file_path.unlink()
                        prefix = data_dir if merge_file_path.is_relative_to(data_dir) else Config.WORK_DIR
                        # Reset to workdir
                        merge_file_path = Path(Config.WORK_DIR) / merge_file_path.relative_to(prefix)
                        raise IOError('Missing Datasource')

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

            logging.warning(f"Merge {str(merge_file_path)} executed successfully. "
                            f"Efficacy={activity['args']['efficacy']}, "
                            f"cloud_ratio={activity['args']['cloudratio']}")

            activity['args']['reused'] = reused

            return activity
        except Exception as e:
            logging.warning(f'{merge_file_path.name} not valid due {str(e)}. Recreating...')

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
            res = merge_processing(str(merge_file_path), mask, band_map=band_map, band=record.band, compute=True,
                                   collection=collection_id, **args, **kwargs)

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


@celery_app.task(queue=Config.QUEUE_PREPARE_CUBE)
def prepare_blend(merges, band_map: dict, reuse_data_cube=None, **kwargs):
    """Receive merges by period and prepare task blend.

    This task aims to prepare celery task definition for blend.
    A blend requires both data set quality band and others bands. In this way, we must group
    these values by temporal resolution and then schedule blend tasks.
    """
    block_size = kwargs.get('block_size')
    quality_band = kwargs['quality_band']

    activities = dict()

    # Prepare map of efficacy/cloud_ratio based in quality merge result
    quality_date_stats = {
        m['date']: (m['args']['efficacy'], m['args']['cloudratio'], m['args']['file'], m['args']['reused'])
        for m in merges if m['band'] == quality_band or quality_band is None  # Quality None means IDT without Mask
    }

    version = merges[0]['args']['version']
    identity_cube = merges[0]['warped_collection_id']

    if reuse_data_cube:
        identity_cube = get_cube_id(reuse_data_cube['name'])
        version = reuse_data_cube['version']

    kwargs['mask'] = kwargs['mask'] or dict()
    bands = [b for b in band_map.keys() if b != kwargs['mask'].get('saturated_band')]

    if 'no_post_process' not in kwargs:
        for period, stats in quality_date_stats.items():
            _, _, quality_file, was_reused = stats

            # Do not apply post-processing on reused data cube since it may be already processed.
            if not was_reused:
                logging.info(f'Applying post-processing in {str(quality_file)}')
                post_processing_quality(quality_file, bands, identity_cube,
                                        period, merges[0]['tile_id'], quality_band, band_map,
                                        version=version, block_size=block_size,
                                        datasets=merges[0]['args']['datasets'])
            else:
                logging.info(f'Skipping post-processing {str(quality_file)}')

    def _is_not_stk(_merge):
        """Control flag to generate cloud mask.

        This function is a utility to dispatch the cloud mask generation only for STK data cubes.
        """
        collection_id = _merge['collection_id']
        fragments = DataCubeFragments(collection_id)
        return _merge['band'] == quality_band and fragments.composite_function not in ('STK', 'LCF')

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
        activity['mask'] = kwargs.get('mask')
        # TODO: Check instance type for backward compatibility
        activity['datasets'] = _merge['args']['datasets']

        # Map efficacy/cloud ratio to the respective merge date before pass to blend
        efficacy, cloudratio, quality_file, _ = quality_date_stats[_merge['date']]
        activity['scenes'][_merge['args']['date']]['efficacy'] = efficacy
        activity['scenes'][_merge['args']['date']]['cloudratio'] = cloudratio

        if _merge['args'].get('reuse_datacube'):
            activity['reuse_datacube'] = _merge['args']['reuse_datacube']

        activity['scenes'][_merge['args']['date']]['ARDfiles'] = {
            _merge['band']: _merge['args']['file']
        }
        if quality_band is not None:
            activity['scenes'][_merge['args']['date']]['ARDfiles'][quality_band] = quality_file

        if _merge['args'].get(DATASOURCE_NAME):
            activity['scenes'][_merge['args']['date']]['ARDfiles'][DATASOURCE_NAME] = _merge['args'][DATASOURCE_NAME]

        activities[_merge['band']] = activity

    if kwargs.get('mask') and kwargs['mask'].get('saturated_band'):
        saturated_mask = kwargs['mask']['saturated_band']

        if saturated_mask not in activities:
            raise RuntimeError(f'Unexpected error: Missing {saturated_mask}')

        reference = activities[saturated_mask]

        for band, activity in activities.items():
            for merge_date, scene in activity['scenes'].items():
                saturated_merge_file = reference['scenes'][merge_date]['ARDfiles'][saturated_mask]
                scene['ARDfiles'][saturated_mask] = saturated_merge_file

    # TODO: Add option to skip histogram.
    if kwargs.get('histogram_matching'):
        ordered_best_efficacy = sorted(quality_date_stats.items(), key=lambda item: item[1][0], reverse=True)

        best_date, (_, _, best_mask_file, _) = ordered_best_efficacy[0]
        dates = map(lambda entry: entry[0], ordered_best_efficacy[1:])

        for date in dates:
            logging.info(f'Applying Histogram Matching: Reference date {best_date}, current {date}...')
            for band, activity in activities.items():
                reference = activities[band]['scenes'][best_date]['ARDfiles'][band]

                if band == quality_band:
                    continue

                source = activity['scenes'][date]['ARDfiles'][band]
                source_mask = activity['scenes'][date]['ARDfiles'][quality_band]
                match_histogram_with_merges(source, source_mask, reference, best_mask_file, block_size=block_size)

    if kwargs.get('reference_day'):
        timeline = list(quality_date_stats.keys())
        ordered_dates = temporal_priority_timeline(kwargs['reference_day'], timeline)
        # TODO: Check factor to generate weights based in len of ordered_dates
        weights = [100 - (idx * 0.01) for idx, _ in enumerate(ordered_dates)]

        for activity in activities.values():
            for idx, date in enumerate(ordered_dates):
                activity['scenes'][date]['efficacy'] = weights[idx]

    # Prepare list of activities to dispatch
    activity_list = list(activities.values())

    datacube = activity_list[0]['datacube']

    # For IDENTITY data cube trigger, just publish
    if DataCubeFragments(datacube).composite_function == 'IDT':
        task = publish.s(list(activities.values()), reuse_data_cube=reuse_data_cube, band_map=band_map, **kwargs)
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
        blends.append(blend.s(activity, band_map, reuse_data_cube=reuse_data_cube, **kwargs))

    # Trigger last blend to execute Clear Observation
    blends.append(blend.s(last_activity, band_map, build_clear_observation=True, reuse_data_cube=reuse_data_cube, **kwargs))

    task = chain(group(blends), publish.s(band_map, reuse_data_cube=reuse_data_cube, **kwargs))
    task.apply_async()
    return blends


@celery_app.task(queue=Config.QUEUE_BLEND_CUBE)
def blend(activity, band_map, build_clear_observation=False, reuse_data_cube=None, **kwargs):
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

    return blend_processing(activity, band_map, kwargs['quality_band'], build_clear_observation,
                            block_size=block_size, reuse_data_cube=reuse_data_cube,
                            apply_valid_range=kwargs.get('apply_valid_range'))


@celery_app.task(queue=Config.QUEUE_PUBLISH_CUBE)
def publish(blends, band_map, quality_band: str, reuse_data_cube=None, **kwargs):
    """Execute publish task and catalog datacube result.

    Args:
        activity - Datacube Activity Model
    """
    if isinstance(blends, Iterable):
        blend_reference = blends[0]
    else:
        blend_reference = blends

    period = blend_reference['period']
    logging.info(f'Executing publish {period}')

    version = blend_reference['version']

    cube: Collection = Collection.query().filter(
        Collection.name == blend_reference['datacube'],
        Collection.version == version
    ).first()
    warped_datacube = blend_reference['warped_datacube']
    tile_id = blend_reference['tile_id']
    reused_cube = blend_reference.get('reuse_datacube')

    # Retrieve which bands to generate quick look
    bands = cube.bands
    band_id_map = {band.id: band.name for band in bands}

    quicklook = cube.quicklook[0]

    quick_look_bands = [band_id_map[quicklook.red], band_id_map[quicklook.green], band_id_map[quicklook.blue]]

    merges = dict()
    blend_files = dict()

    composite_function = DataCubeFragments(cube.name).composite_function

    quality_blend = dict(efficacy=100, cloudratio=0)

    for blend_result in blends:
        if composite_function != 'IDT':
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
            merges[merge_date]['empty'] = definition.get('empty', False)

        if blend_result['band'] == quality_band:
            quality_blend = blend_result

    _blend_result = []
    cube_geom_table = cube.grs.geom_table
    srid_column = get_srid_column(cube_geom_table.c)
    srid = None
    result = db.session.query(srid_column.label('srid')).first()
    if result is not None:
        srid = result.srid

    if composite_function != 'IDT':
        cloudratio = quality_blend['cloudratio']

        # Generate quick looks for cube scenes
        _blend_result = publish_datacube(cube, quick_look_bands, tile_id, period, blend_files, cloudratio, reuse_data_cube=reuse_data_cube, srid=srid, **kwargs)

    # Generate quick looks of irregular cube
    wcube = Collection.query().filter(Collection.name == warped_datacube, Collection.version == version).first()

    _merge_result = dict()

    if not reused_cube:
        for merge_date, definition in merges.items():
            if definition.get('empty') and definition['empty']:
                # Empty data cubes, Keep only composite item
                clear_merge(merge_date, definition)
                continue

            _merge_result[merge_date] = publish_merge(quick_look_bands, wcube, tile_id, merge_date, definition,
                                                      reuse_data_cube=reuse_data_cube, srid=srid, data_dir=kwargs.get('data_dir'))

        try:
            db.session.commit()
        except:
            db.session.rollback()

    return _blend_result, _merge_result
