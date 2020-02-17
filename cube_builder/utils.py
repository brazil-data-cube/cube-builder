# Python Native
from datetime import datetime
import logging
import os
# 3rdparty
from numpngw import write_png
from rasterio import Affine, MemoryFile
from rasterio.warp import reproject, Resampling
import numpy
import rasterio
# BDC Scripts
from bdc_db.models import Asset, Band, CollectionItem, db
from .config import Config
from .models import Activity


def get_or_create_model(model_class, defaults=None, **restrictions):
    """
    Utility method for looking up an object with the given restrictions, creating
    one if necessary.
    Args:
        model_class (BaseModel) - Base Model of Brazil Data Cube DB
        defaults (dict) - Values to fill out model instance
        restrictions (dict) - Query Restrictions
    Returns:
        BaseModel Retrieves model instance
    """

    instance = db.session.query(model_class).filter_by(**restrictions).first()

    if instance:
        return instance, False

    params = dict((k, v) for k, v in restrictions.items())

    params.update(defaults or {})
    instance = model_class(**params)

    db.session.add(instance)

    return instance, True


def get_or_create_activity(cube: str, warped: str, activity_type: str, scene_type: str, band: str, period: str, activity_date: str, **parameters):
    defaults = dict(
        band=band,
        collection_id=cube,
        warped_collection_id=warped,
        activity_type=activity_type,
        tags=parameters.get('tags', []),
        status='CREATED',
        date=activity_date,
        period=period,
        scene_type=scene_type,
        args=parameters
    )

    where = dict(
        band=band,
        collection_id=cube,
        period=period,
        date=activity_date
    )

    return get_or_create_model(Activity, defaults=defaults, **where)


def build_datacube_name(datacube, func):
    return '{}{}'.format(datacube[:-3], func)


def merge(warped_datacube, tile_id, assets, cols, rows, period, **kwargs):
    datacube = kwargs['datacube']
    nodata = kwargs.get('nodata', -9999)
    xmin = kwargs.get('xmin')
    ymax = kwargs.get('ymax')
    dataset = kwargs.get('dataset')
    band = assets[0]['band']
    merge_date = kwargs.get('date')
    resx, resy = kwargs.get('resx'), kwargs.get('resy')

    formatted_date = datetime.strptime(merge_date, '%Y-%m-%d').strftime('%Y%m%d')

    srs = kwargs.get('srs', '+proj=aea +lat_1=10 +lat_2=-40 +lat_0=0 +lon_0=-50 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    merge_name = '{}-{}-{}-{}'.format(warped_datacube, tile_id, formatted_date, band)

    folder_name = warped_datacube.replace('_WARPED', '')

    merged_file = os.path.join(Config.DATA_DIR, 'Repository/Warped/{}/{}/{}/{}.tif'.format(folder_name, tile_id, period, merge_name))

    transform = Affine(resx, 0, xmin, 0, -resy, ymax)

    # Quality band is resampled by nearest, other are bilinear
    if band == 'quality':
        resampling = Resampling.nearest

        raster = numpy.zeros((rows, cols,), dtype=numpy.uint16)
        raster_merge = numpy.zeros((rows, cols,), dtype=numpy.uint16)
        raster_mask = numpy.ones((rows, cols,), dtype=numpy.uint16)
        nodata = 0
    else:
        resampling = Resampling.bilinear
        raster = numpy.zeros((rows, cols,), dtype=numpy.int16)
        raster_merge = numpy.full((rows, cols,), fill_value=nodata, dtype=numpy.int16)

    count = 0
    template = None
    for asset in assets:
        count += 1
        with rasterio.Env(CPL_CURL_VERBOSE=False):
            with rasterio.open(asset['link']) as src:
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': srs,
                    'transform': transform,
                    'width': cols,
                    'height': rows
                })

                source_nodata = 0

                if src.profile['nodata'] is not None:
                    source_nodata = src.profile['nodata']
                elif 'LC8SR' in dataset and band != 'quality':
                    source_nodata = nodata

                kwargs.update({
                    'nodata': source_nodata
                })

                with MemoryFile() as mem_file:
                    with mem_file.open(**kwargs) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=raster,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=srs,
                            src_nodata=source_nodata,
                            dst_nodata=nodata,
                            resampling=resampling)

                        if band != 'quality':
                            valid_data_scene = raster[raster != nodata]
                            raster_merge[raster != nodata] = valid_data_scene.reshape(numpy.size(valid_data_scene))
                        else:
                            raster_merge = raster_merge + raster * raster_mask
                            raster_mask[raster != nodata] = 0

                        if template is None:
                            template = dst.profile
                            # Ensure type is >= int16

                            if band != 'quality':
                                template['dtype'] = 'int16'
                                template['nodata'] = nodata

    # Evaluate cloud cover and efficacy if band is quality
    efficacy = 0
    cloudratio = 100
    if band == 'quality':
        raster_merge, efficacy, cloudratio = getMask(raster_merge, dataset)
        template.update({'dtype': 'uint16'})

    target_dir = os.path.dirname(merged_file)
    os.makedirs(target_dir, exist_ok=True)

    template.update({
        'compress': 'LZW',
        'tiled': True,
        "interleave": "pixel",
    })

    with rasterio.open(merged_file, 'w', **template) as merge_dataset:
        merge_dataset.nodata = nodata
        merge_dataset.write_band(1, raster_merge)
        merge_dataset.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
        merge_dataset.update_tags(ns='rio_overview', resampling='nearest')

    return dict(
        band=band,
        file=merged_file,
        efficacy=efficacy,
        cloudratio=cloudratio,
        dataset=dataset,
        resolution=resx,
        period=period,
        date='{}{}'.format(merge_date, dataset),
        datacube=datacube,
        tile_id=tile_id,
        warped_datacube=warped_datacube,
        nodata=nodata
    )


def blend(activity):
    # Assume that it contains a band and quality band
    numscenes = len(activity['scenes'])

    band = activity['band']

    nodata = activity.get('nodata', -9999)

    # Get basic information (profile) of input files
    keys = list(activity['scenes'].keys())

    filename = activity['scenes'][keys[0]]['ARDfiles'][band]

    with rasterio.open(filename) as src:
        profile = src.profile
        tilelist = list(src.block_windows())

    # Order scenes based in efficacy/resolution
    mask_tuples = []

    for key in activity['scenes']:
        scene = activity['scenes'][key]
        efficacy = int(scene['efficacy'])
        resolution = int(scene['resolution'])
        mask_tuples.append((100. * efficacy / resolution, key))

    # Open all input files and save the datasets in two lists, one for masks and other for the current band.
    # The list will be ordered by efficacy/resolution
    masklist = []
    bandlist = []
    efficacy = 0
    for m in sorted(mask_tuples, reverse=True):
        key = m[1]
        efficacy = m[0]
        scene = activity['scenes'][key]

        filename = scene['ARDfiles']['quality']

        try:
            masklist.append(rasterio.open(filename))
        except BaseException as e:
            raise IOError('FileError while opening {} - {}'.format(filename, e))

        filename = scene['ARDfiles'][band]

        try:
            bandlist.append(rasterio.open(filename))
        except BaseException as e:
            raise IOError('FileError while opening {} - {}'.format(filename, e))

    # Build the raster to store the output images.
    width = profile['width']
    height = profile['height']

    # STACK will be generated in memory
    stack_raster = numpy.zeros((height, width), dtype=profile['dtype'])
    mask_raster = numpy.ones((height, width), dtype=profile['dtype'])

    datacube = activity.get('datacube')
    period = activity.get('period')
    tile_id = activity.get('tile_id')
    output_name = '{}-{}-{}-{}'.format(datacube, tile_id, period, band)

    #
    # MEDIAN will be generated in local disk
    medianfile = os.path.join(Config.DATA_DIR, 'Repository/Mosaic/{}/{}/{}/{}.tif'.format(
        datacube, tile_id, period, output_name))

    stack_datacube = build_datacube_name(datacube, 'STK')
    output_name = output_name.replace(datacube, stack_datacube)

    stack_file = os.path.join(Config.DATA_DIR, 'Repository/Mosaic/{}/{}/{}/{}.tif'.format(
        stack_datacube, tile_id, period, output_name))

    os.makedirs(os.path.dirname(medianfile), exist_ok=True)
    os.makedirs(os.path.dirname(stack_file), exist_ok=True)

    mediandataset = rasterio.open(medianfile, 'w', **profile)

    for _, window in tilelist:
        # Build the stack to store all images as a masked array. At this stage the array will contain the masked data
        stackMA = numpy.ma.zeros((numscenes, window.height, window.width), dtype=numpy.int16)

        notdonemask = numpy.ones(shape=(window.height, window.width), dtype=numpy.bool_)

        # For all pair (quality,band) scenes
        for order in range(numscenes):
            ssrc = bandlist[order]
            msrc = masklist[order]
            raster = ssrc.read(1, window=window)
            mask = msrc.read(1, window=window)
            mask[mask != 1] = 0

            mask[raster == nodata] = 0

            # True => nodata
            bmask = numpy.invert(mask.astype(numpy.bool_))

            # Use the mask to mark the fill (0) and cloudy (2) pixels
            stackMA[order] = numpy.ma.masked_where(bmask, raster)

            # Evaluate the STACK image
            # Pixels that have been already been filled by previous rasters will be masked in the current raster
            mask_raster[window.row_off: window.row_off + window.height, window.col_off: window.col_off + window.width] *= bmask.astype(profile['dtype'])

            raster[raster == nodata] = 0
            todomask = notdonemask * numpy.invert(bmask)
            notdonemask = notdonemask * bmask
            stack_raster[window.row_off: window.row_off + window.height, window.col_off: window.col_off + window.width] += (todomask * raster.astype(profile['dtype']))

        median_raster = numpy.ma.median(stackMA, axis=0).data
        median_raster[notdonemask.astype(numpy.bool_)] = nodata

        mediandataset.write(median_raster.astype(profile['dtype']), window=window, indexes=1)

    stack_raster[mask_raster.astype(numpy.bool_)] = nodata
    # Close all input dataset
    for order in range(numscenes):
        bandlist[order].close()
        masklist[order].close()

    # Evaluate cloudcover
    cloudcover = 100. * ((height * width - numpy.count_nonzero(stack_raster)) / (height * width))

    if band != 'quality':
        mediandataset.nodata = nodata

    # # Close and upload the MEDIAN dataset
    mediandataset.close()

    profile.update({
        'compress': 'LZW',
        'tiled': True,
        "interleave": "pixel",
    })

    with rasterio.open(medianfile, 'r+', **profile) as ds_median:
        ds_median.nodata = nodata
        ds_median.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
        ds_median.update_tags(ns='rio_overview', resampling='nearest')

    with rasterio.open(stack_file, 'w', **profile) as stack_dataset:
        stack_dataset.nodata = nodata
        stack_dataset.write_band(1, stack_raster)
        stack_dataset.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
        stack_dataset.update_tags(ns='rio_overview', resampling='nearest')

    activity['efficacy'] = efficacy
    activity['cloudratio'] = cloudcover
    activity['blends'] = {
        "MED": medianfile,
        "STK": stack_file
    }

    return activity


def publish_datacube(cube, bands, datacube, tile_id, period, scenes, cloudratio):
    item_id = '{}_{}_{}'.format(cube.id, tile_id, period)
    start_date, end_date = period.split('_')

    cube_bands = Band.query().filter(Band.collection_id == cube.id).all()
    raster_size_schemas = cube.raster_size_schemas

    for composite_function in ['MED', 'STK']:
        _datacube = build_datacube_name(datacube, composite_function)

        quick_look_name = '{}-{}-{}'.format(_datacube, tile_id, period)

        quick_look_relpath = 'Repository/Mosaic/{}/{}/{}/{}'.format(
            _datacube, tile_id, period, quick_look_name
        )
        quick_look_file = os.path.join(
            Config.DATA_DIR,
            quick_look_relpath
        )

        ql_files = []
        for band in bands:
            ql_files.append(scenes[band][composite_function])

        generate_quick_look(quick_look_file, ql_files)

        Asset.query().filter(Asset.collection_item_id == item_id).delete()

        CollectionItem.query().filter(CollectionItem.id == item_id).delete()

        with db.session.begin_nested():
            CollectionItem(
                id=item_id,
                collection_id=cube.id,
                grs_schema_id=cube.grs_schema_id,
                tile_id=tile_id,
                item_date=start_date,
                composite_start=start_date,
                composite_end=end_date,
                quicklook=quick_look_file,
                cloud_cover=cloudratio,
                scene_type=composite_function,
                compressed_file=None
            ).save(commit=False)

            for band in scenes:
                if band == 'quality':
                    continue

                band_model = next(filter(lambda b: b.common_name == band, cube_bands))

                # Band does not exists on model
                if not band_model:
                    logging.warning('Band {} of {} does not exist on database'.format(band, cube.id))
                    continue

                asset_relative_path = scenes[band][composite_function].replace(Config.DATA_DIR, '')

                Asset(
                    collection_id=cube.id,
                    band_id=band_model.id,
                    grs_schema_id=cube.grs_schema_id,
                    tile_id=tile_id,
                    collection_item_id=item_id,
                    url='{}'.format(asset_relative_path),
                    source=None,
                    raster_size_x=raster_size_schemas.raster_size_x,
                    raster_size_y=raster_size_schemas.raster_size_y,
                    raster_size_t=1,
                    chunk_size_x=raster_size_schemas.chunk_size_x,
                    chunk_size_y=raster_size_schemas.chunk_size_y,
                    chunk_size_t=1
                ).save(commit=False)

        db.session.commit()

    return quick_look_file


def publish_merge(bands, datacube, dataset, tile_id, period, date, scenes):
    item_id = '{}_{}_{}'.format(datacube.id, tile_id, period)
    quick_look_name = '{}-{}-{}'.format(datacube.id, tile_id, date)
    quick_look_file = os.path.join(
        Config.DATA_DIR,
        'Repository/Warped/{}/{}/{}/{}'.format(
            datacube.id.replace('_WARPED', ''), tile_id, period, quick_look_name
        )
    )

    cube_bands = Band.query().filter(Band.collection_id == datacube.id).all()
    raster_size_schemas = datacube.raster_size_schemas

    ql_files = []
    for band in bands:
        ql_files.append(scenes['ARDfiles'][band])

    generate_quick_look(quick_look_file, ql_files)

    Asset.query().filter(Asset.collection_item_id == item_id).delete()

    CollectionItem.query().filter(CollectionItem.id == item_id).delete()

    with db.session.begin_nested():
        CollectionItem(
            id=item_id,
            collection_id=datacube.id,
            grs_schema_id=datacube.grs_schema_id,
            tile_id=tile_id,
            item_date=date,
            composite_start=date,
            composite_end=period.split('_')[-1],
            quicklook=quick_look_file,
            cloud_cover=scenes.get('cloudratio', 0),
            scene_type='WARPED',
            compressed_file=None
        ).save(commit=False)

        for band in scenes['ARDfiles']:
            band_model = next(filter(lambda b: b.common_name == band, cube_bands))

            # Band does not exists on model
            if not band_model:
                logging.warning('Band {} of {} does not exist on database'.format(band, datacube.id))
                continue

            asset_relative_path = scenes['ARDfiles'][band].replace(Config.DATA_DIR, '')

            Asset(
                collection_id=datacube.id,
                band_id=band_model.id,
                grs_schema_id=datacube.grs_schema_id,
                tile_id=tile_id,
                collection_item_id=item_id,
                url='{}'.format(asset_relative_path),
                source=None,
                raster_size_x=raster_size_schemas.raster_size_x,
                raster_size_y=raster_size_schemas.raster_size_y,
                raster_size_t=1,
                chunk_size_x=raster_size_schemas.chunk_size_x,
                chunk_size_y=raster_size_schemas.chunk_size_y,
                chunk_size_t=1
            ).save(commit=False)

    db.session.commit()

    return quick_look_file


def generate_quick_look(file_path, qlfiles):
    with rasterio.open(qlfiles[0]) as src:
        profile = src.profile

    numlin = 768
    numcol = int(float(profile['width'])/float(profile['height'])*numlin)
    image = numpy.ones((numlin,numcol,len(qlfiles),), dtype=numpy.uint8)
    pngname = '{}.png'.format(file_path)

    nb = 0
    for file in qlfiles:
        with rasterio.open(file) as src:
            raster = src.read(1, out_shape=(numlin, numcol))

            # Rescale to 0-255 values
            nodata = raster <= 0
            if raster.min() != 0 or raster.max() != 0:
                raster = raster.astype(numpy.float32)/10000.*255.
                raster[raster > 255] = 255
            image[:, :, nb] = raster.astype(numpy.uint8) * numpy.invert(nodata)
            nb += 1

    write_png(pngname, image, transparent=(0, 0, 0))
    return pngname


def getMask(raster, dataset):
    from skimage import morphology
    # Output Cloud Mask codes
    # 0 - fill
    # 1 - clear data
    # 0 - cloud
    if 'LC8SR' in dataset:
        # Input pixel_qa codes
        fill = 1 				# warped images have 0 as fill area
        # terrain = 2					# 0000 0000 0000 0010
        radsat = 4+8				# 0000 0000 0000 1100
        cloud = 16+32+64			# 0000 0000 0110 0000
        shadow = 128+256			# 0000 0001 1000 0000
        snowice = 512+1024			# 0000 0110 0000 0000
        cirrus = 2048+4096			# 0001 1000 0000 0000

        # Start with a zeroed image imagearea
        imagearea = numpy.zeros(raster.shape, dtype=numpy.bool_)
        # Mark with True the pixels that contain valid data
        imagearea = imagearea + raster > fill
        # Create a notcleararea mask with True where the quality criteria is as follows
        notcleararea = (raster & radsat > 4) + \
            (raster & cloud > 64) + \
            (raster & shadow > 256) + \
            (raster & snowice > 512) + \
            (raster & cirrus > 4096)

        strel = morphology.selem.square(6)
        notcleararea = morphology.binary_dilation(notcleararea,strel)
        morphology.remove_small_holes(notcleararea, area_threshold=80, connectivity=1, in_place=True)

        # Clear area is the area with valid data and with no Cloud or Snow
        cleararea = imagearea * numpy.invert(notcleararea)
        # Code the output image rastercm as the output codes
        rastercm = (2*notcleararea + cleararea).astype(numpy.uint16)  # .astype(numpy.uint8)

    elif dataset == 'MOD13Q1' or dataset == 'MYD13Q1':
        # MOD13Q1 Pixel Reliability !!!!!!!!!!!!!!!!!!!!
        # Note that 1 was added to this image in downloadModis because of warping
        # Rank/Key Summary QA 		Description
        # -1 		Fill/No Data 	Not Processed
        # 0 		Good Data 		Use with confidence
        # 1 		Marginal data 	Useful, but look at other QA information
        # 2 		Snow/Ice 		Target covered with snow/ice
        # 3 		Cloudy 			Target not visible, covered with cloud
        fill = 0 	# warped images have 0 as fill area
        lut = numpy.array([0, 1, 1, 2, 2], dtype=numpy.uint8)
        rastercm = numpy.take(lut, raster+1).astype(numpy.uint8)

    elif 'S2SR' in dataset:
        # S2 sen2cor - The generated classification map is specified as follows:
        # Label Classification
        #  0		NO_DATA
        #  1		SATURATED_OR_DEFECTIVE
        #  2		DARK_AREA_PIXELS
        #  3		CLOUD_SHADOWS
        #  4		VEGETATION
        #  5		NOT_VEGETATED
        #  6		WATER
        #  7		UNCLASSIFIED
        #  8		CLOUD_MEDIUM_PROBABILITY
        #  9		CLOUD_HIGH_PROBABILITY
        # 10		THIN_CIRRUS
        # 11		SNOW
        # 0 1 2 3 4 5 6 7 8 9 10 11
        lut = numpy.array([0,0,2,2,1,1,1,2,2,2,1, 1],dtype=numpy.uint8)
        rastercm = numpy.take(lut,raster).astype(numpy.uint8)

    elif dataset == 'CB4_AWFI' or dataset == 'CB4_MUX':
        # Key 		Summary QA 		Description
        # 0 		Fill/No Data 	Not Processed
        # 127 		Good Data 		Use with confidence
        # 255 		Cloudy 			Target not visible, covered with cloud
        fill = 0 		# warped images have 0 as fill area
        lut = numpy.zeros(256,dtype=numpy.uint8)
        lut[127] = 1
        lut[255] = 2
        rastercm = numpy.take(lut, raster).astype(numpy.uint8)

    totpix = rastercm.size
    clearpix = numpy.count_nonzero(rastercm == 1)
    cloudpix = numpy.count_nonzero(rastercm == 2)
    imagearea = clearpix+cloudpix
    cloudratio = 100
    if imagearea != 0:
        cloudratio = round(100.*cloudpix/imagearea, 1)
    efficacy = round(100.*clearpix/totpix, 2)

    return rastercm.astype(numpy.uint16), efficacy, cloudratio
