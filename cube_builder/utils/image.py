#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#

"""Define a utility to validate merge images."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Union
from urllib.parse import urlparse

import numpy
import rasterio
from rasterio._warp import Affine
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from sqlalchemy.engine.result import ResultProxy, RowProxy

from ..config import Config

LANDSAT_BANDS = dict(
    int16=['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7', 'evi', 'ndvi'],
    uint16=['pixel_qa']
)


def validate(row: RowProxy):
    """Validate each merge result."""
    url = row.link.replace('chronos.dpi.inpe.br:8089/datastore', 'www.dpi.inpe.br/newcatalog/tmp')

    errors = list()

    if 'File' in row.traceback:
        try:
            with rasterio.open('/vsicurl/{}'.format(url), 'r') as data_set:
                logging.debug('File {} ok'.format(url))

                if row.collection_id.startswith('LC8'):
                    data_type = data_set.meta.get('dtype')

                    band_dtype = LANDSAT_BANDS.get(data_type)

                    if band_dtype is None:
                        errors.append(
                            dict(
                                message='Band {} mismatch with default Landsat 8'.format(row.band),
                                band=row.band,
                                file=url
                            )
                        )

                    file_name = Path(url).stem

                    band_name = file_name.split('_')[8]

                    if band_name not in band_dtype:
                        errors.append(
                            dict(
                                message='Band {} should be {}'.format(row.band, data_type),
                                band=row.band,
                                file=url
                            )
                        )

        except rasterio.RasterioIOError as e:
            if not row.traceback:
                errors.append(dict(message=f'File not found or invalid. ({url})', band=row.band,
                                   file=url, filename=_file_name(url)))

    return row, errors


def _file_name(url: str) -> str:
    parsed = urlparse(url)
    # IF SAFE: parent
    if '.SAFE' in parsed.path:
        safe_pos = parsed.path.index('.SAFE') + 5
        abs_safe_folder = parsed.path[:safe_pos]
        scene_id = os.path.basename(os.path.dirname(abs_safe_folder))
        return scene_id

    return os.path.basename(parsed.path)


def validate_merges(images: ResultProxy, num_threads: int = Config.MAX_THREADS_IMAGE_VALIDATOR) -> dict:
    """Validate each merge retrieved from ``Activity.list_merge_files``.

    Args:
        images: Activity merge images
        num_threads: Concurrent processes to validate
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = executor.map(validate, images)

        output = dict()
        error_map = dict()

        for row, errors in futures:
            if row is None:
                continue

            output.setdefault(row.date, dict())
            output[row.date].setdefault('bands', dict())
            output[row.date].setdefault('errors', list())
            output[row.date].setdefault('collections', set())

            output[row.date]['collections'].add(row.data_set)

            output[row.date]['file'] = row.file
            output[row.date]['errors'].extend(errors)
            if row.traceback and row.date not in error_map:
                output[row.date]['errors'].append(dict(message=row.traceback, band=row.band,
                                                       filename=_file_name(row.link)))
                error_map[row.date] = True

            output[row.date]['bands'].setdefault(row.band, list())
            output[row.date]['bands'][row.band].append(row.link)

        for element in output.values():
            element['collections'] = list(element['collections'])

        return output


def create_empty_raster(location: str, proj4: str, dtype: str, xmin: float, ymax: float,
                        resolution: List[float], dist: List[float], nodata: float, cog=True):
    """Create an data set filled out with nodata.

    This method aims to solve the problem to generate an empty scene to make sure in order to
    follow the data cube timeline.

    Args:
        location (str): Path where file will be generated.
        proj4 (str): Proj4 with Coordinate Reference System.
        dtype (str): Data type
        xmin (float): Image minx (Related to geotransform)
        ymax (float): Image ymax
        resolution (List[float]): Pixel resolution (X, Y)
        dist (List[float]): The distance of X, Y  (Scene offset)
        nodata (float): Scene nodata.
        cog (bool): Flag to generate datacube. Default is `True`.
    """
    resx, resy = resolution
    distx, disty = dist

    cols = round(distx / resx)
    rows = round(disty / resy)

    new_res_x = distx / cols
    new_res_y = disty / rows

    transform = Affine(new_res_x, 0, xmin, 0, -new_res_y, ymax)

    options = dict(
        width=cols,
        height=rows,
        nodata=nodata,
        crs=proj4,
        transform=transform,
        count=1
    )

    ds = SmartDataSet(str(location), mode='w', dtype=dtype, driver='GTiff', **options)

    ds.close()

    if cog:
        generate_cogs(str(location), str(location))

    return str(location)


def match_histogram_with_merges(source: str, source_mask: str, reference: str, reference_mask: str, **kwargs):
    """Normalize the source image histogram with reference image.

    This functions implements the `skimage.exposure.match_histograms`, which consists in the manipulate the pixels of an
    input image and match the histogram with the reference image.

    See more in `Histogram Matching <https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html>`_.

    Note:
        It overwrites the source file.

    Args:
        source (str): Path to the rasterio data set file
        source_mask (str): Path to the rasterio data set file
        reference (str): Path to the rasterio data set file
        reference_mask (str): Path to the rasterio data set file
    """
    from skimage.exposure import match_histograms as _match_histograms

    block_size = kwargs.get('block_size')

    with rasterio.open(source) as source_data_set, rasterio.open(source_mask) as source_mask_data_set:
        source_arr = source_data_set.read(1, masked=True)
        source_mask_arr = source_mask_data_set.read(1)
        source_nodata = kwargs.get('source_nodata', source_mask_data_set.nodata)
        source_options = source_data_set.profile.copy()

    with rasterio.open(reference) as reference_data_set, rasterio.open(reference_mask) as reference_mask_data_set:
        reference_arr = reference_data_set.read(1, masked=True)
        reference_nodata = kwargs.get('reference_nodata', reference_mask_data_set.nodata)
        reference_mask_arr = reference_mask_data_set.read(1)

    intersect_mask = numpy.logical_and(
        source_mask_arr != source_nodata,  # CHECK: Use only valid data? numpy.isin(source_mask_arr, [0, 1, 3]),
        reference_mask_arr != reference_nodata,   # CHECK: Use only valid data? numpy.isin(reference_mask_arr, [0, 1, 3]),
    )

    valid_positions = numpy.where(intersect_mask)

    if valid_positions and len(valid_positions[0]) == 0:
        return

    intersected_source_arr = source_arr[valid_positions]
    intersected_reference_arr = reference_arr[valid_positions]

    histogram = _match_histograms(intersected_source_arr, intersected_reference_arr)

    histogram = numpy.round(histogram).astype(source_options['dtype'])

    source_arr[valid_positions] = histogram

    save_as_cog(str(source), source_arr, block_size=block_size, mode='w', **source_options)


def radsat_extract_bits(bit_value: Union[int, numpy.ndarray], bit_start: int, bit_end: Optional[int] = None):
    """Extract bitwise values from image.

    This method uses the bitwise operation to identify pixel saturation.
    According to the document `LaSRC Product Guide <https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1368_L8_C1-LandSurfaceReflectanceCode-LASRC_ProductGuide-v3.pdf>`_,
    the Landsat Radiometric Saturation Quality Assessment Band (radsat_qa) is a bit
    packed representation of which sensor bands were saturated during data sensing capture.
    The value 1 represents saturated value while 0 is valid data.
    For Landsat-8, the following table represents pixels saturation::

        Bit    Bit Value    Description
          0        1        Data Fill Flag
          1        2        Band 1 Data Saturation Flag
          2        4        Band 2 Data Saturation Flag
          3        8        Band 3 Data Saturation Flag
          4       16        Band 4 Data Saturation Flag
          5       32        Band 5 Data Saturation Flag
          6       64        Band 6 Data Saturation Flag
          7      128        Band 7 Data Saturation Flag
          8      256        Band 8 Data Saturation Flag
          9      512        Band 9 Data Saturation Flag
         10     1024        Band 10 Data Saturation Flag
         11     2048        Band 11 Data Saturation Flag

    Example:
        >>> from cube_builder.utils.image import radsat_extract_bits
        >>> # Represents band 10 (1024) and band 1 (2) is saturated.
        >>> # Check if any band is saturated
        >>> radsat_extract_bits(1026, 1, 7)
        1
        >>> # You can also pass the numpy array
        >>> # radsat_extract_bits(numpy.random.randint(0, 1028, size=(100, 100)), 1, 7)
    """
    if bit_end is None:
        bit_end = bit_start

    mask_size = (1 + bit_end) - bit_start
    mask = (1 << mask_size) - 1

    res = (bit_value >> bit_start) & mask

    return res


def check_file_integrity(file_path: Union[str, Path], read_bytes: bool = False) -> bool:
    """Check Raster File integrity.

    Args:
        file_path (str|Path): Path to the raster file
        read_bytes (bool): Read raster band entire. Default is False
    """
    try:
        with rasterio.open(file_path) as data_set:
            if read_bytes:
                _ = data_set.read(1)
            return True
    except (rasterio.RasterioIOError, Exception):
        return False


def save_as_cog(destination: str, raster, mode='w', tags=None, block_size=None, **profile):
    """Save the raster file as Cloud Optimized GeoTIFF.

    See Also:
        Cloud Optimized GeoTiff https://gdal.org/drivers/raster/cog.html

    Args:
        destination: Path to store the data set.
        raster: Numpy raster values to persist in disk
        mode: Default rasterio mode. Default is 'w' but you also can set 'r+'.
        tags: Tag values (Dict[str, str]) to write on dataset.
        **profile: Rasterio profile values to add in dataset.
    """
    with rasterio.open(str(destination), mode, **profile) as dataset:
        if profile.get('nodata'):
            dataset.nodata = profile['nodata']

        dataset.write_band(1, raster)

        if tags:
            dataset.update_tags(**tags)

    generate_cogs(str(destination), str(destination), block_size=block_size)


def generate_cogs(input_data_set_path, file_path, profile='deflate', block_size=None, profile_options=None, **options):
    """Generate Cloud Optimized GeoTIFF files (COG).

    Args:
        input_data_set_path (str) - Path to the input data set
        file_path (str) - Target data set filename
        profile (str) - A COG profile based in `rio_cogeo.profiles`.
        profile_options (dict) - Custom options to the profile.
        block_size (int) - Custom block size.

    Returns:
        Path to COG.
    """
    if profile_options is None:
        profile_options = dict()

    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)

    if block_size:
        output_profile["blockxsize"] = block_size
        output_profile["blockysize"] = block_size

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )

    cog_translate(
        str(input_data_set_path),
        str(file_path),
        output_profile,
        config=config,
        in_memory=False,
        quiet=True,
        **options,
    )
    return str(file_path)


class SmartDataSet:
    """Defines utility class to auto close rasterio data set.

    This class is class helper to avoid memory leak of opened data set in memory.
    """

    def __init__(self, file_path: str, mode='r', tags=None, **properties):
        """Initialize SmartDataSet definition and open rasterio data set."""
        self.path = Path(file_path)
        self.mode = mode
        self.dataset = rasterio.open(file_path, mode=mode, **properties)
        self.tags = tags
        self.mode = mode

    def __del__(self):
        """Close dataset on delete object."""
        self.close()

    def __enter__(self):
        """Use data set context with operator ``with``."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close data set on exit with clause."""
        self.close()

    def close(self):
        """Close rasterio data set."""
        if not self.dataset.closed:
            logging.debug('Closing dataset {}'.format(str(self.path)))

            if self.mode == 'w' and self.tags:
                self.dataset.update_tags(**self.tags)

            self.dataset.close()


def extract_qa_bits(band_data, bit_location):
    """Retrieve the bit information from given position.

    Args:
        band_data (int|numpy.ma.masked_array) - The QA Raster Data
        bit_location (int) - The band bit value
    """
    return band_data & (1 << bit_location)


NO_CONFIDENCE = 0
LOW = 1  # 0b01
MEDIUM = RESERVED = 2  # 0b10
HIGH = 3  # 0b11


class QAConfidence:
    """Type for Quality Assessment definition for Landsat Collection 2.

    These properties will be evaluated using Python Virtual Machine like::

        # Define that will discard all cloud values which has confidence greater or equal MEDIUM.
        qa = QAConfidence(cloud='cloud >= MEDIUM', cloud_shadow=None, cirrus=None, snow=None)
    """

    cloud: Union[str, None]
    """Represent the Cloud Confidence."""
    cloud_shadow: Union[str, None]
    """Represent the Cloud Shadow Confidence."""
    cirrus: Union[str, None]
    """Represent the Cirrus."""
    snow: Union[str, None]
    """Represent the Snow/Ice."""
    oli: Union[bool, numpy.ndarray]
    """Flag to identify OLI/TIRS Satellite."""

    def __init__(self, cloud=None, cloud_shadow=None, cirrus=None, snow=None, oli=None):
        """Build QA Confidence object."""
        self.cloud = cloud
        self.cloud_shadow = cloud_shadow
        self.cirrus = cirrus
        self.snow = snow
        self.oli = oli

    def apply(self, data):
        """Apply the Bit confidence to the Quality Assessment mask.

        Args:
            data (numpy.ma.MaskedArray): The Masked raster QA Pixel

        Returns:
            numpy.ma.MaskedArray - The masked pixels with satisfy the confidence attributes.
        """
        from .interpreter import execute

        # Define the context variables available for cloud confidence processing
        ctx = dict(
            NO_CONFIDENCE=NO_CONFIDENCE,
            LOW=LOW,
            MEDIUM=MEDIUM,
            RESERVED=RESERVED,
            HIGH=HIGH
        )

        def _invoke(conf, context_var, start_offset, end_offset, qa):
            var_name = f'_{context_var}'
            expression = f'{var_name} = {conf}'
            array = (qa >> start_offset) - ((qa >> end_offset) << 2)
            ctx[context_var] = array

            _res = execute(expression, context=ctx)
            res = _res[var_name]

            return numpy.ma.masked_where(numpy.ma.getdata(res), qa)

        if self.cloud:
            data = _invoke(self.cloud, 'cloud', 8, 10, data)
        if self.cloud_shadow:
            data = _invoke(self.cloud_shadow, 'cloud_shadow', 10, 12, data)
        if self.snow:
            data = _invoke(self.snow, 'snow', 12, 14, data)

        if self.cirrus:
            if isinstance(self.oli, bool):
                self.oli = numpy.full(data.shape,
                                      dtype=numpy.bool_,
                                      fill_value=self.oli)

            oli_pixels = data[self.oli]
            res = _invoke(self.cirrus, 'cirrus', 14, 16, oli_pixels)

            data[self.oli] = res

        return data


def get_qa_mask(data: numpy.ma.masked_array,
                clear_data: List[float] = None,
                not_clear_data: List[float] = None,
                nodata: float = None,
                confidence: QAConfidence = None) -> numpy.ma.masked_array:
    """Extract Quality Assessment Bits from Landsat Collection 2 Level-2 products.

    This method uses the bitwise operation to extract bits according to the document
    `Landsat 8 Collection 2 (C2) Level 2 Science Product (L2SP) Guide <https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1619_Landsat8-C2-L2-ScienceProductGuide-v2.pdf>`_, page 13.

    Note:
        This method supports the products Landsat-4 (or more) Collection 2 Science products.
        Please take a look into :class:`cube_builder.utils.image.QAConfidence` if you are dealing
        with multiple sensors.

    Example:
        >>> import numpy
        >>> from cube_builder.utils.image import QAConfidence, get_qa_mask

        >>> mid_cloud_confidence = QAConfidence(cloud='cloud == MEDIUM', cloud_shadow=None, cirrus=None, snow=None, oli=True)
        >>> clear = [6, 7]  # Clear and Water
        >>> not_clear = [1, 2, 3, 4]  # Dilated Cloud, Cirrus, Cloud, Cloud Shadow
        >>> get_qa_mask(numpy.ma.array([22080], dtype=numpy.int16, fill_value=1),
        ...             clear_data=clear, not_clear_data=not_clear,
        ...             nodata=1, confidence=mid_cloud_confidence)
        masked_array(data=[--],
                     mask=[ True],
               fill_value=1,
                    dtype=int16)
        >>> # When no cloud confidence set, this value will be Clear since Cloud Pixel is off.
        >>> get_qa_mask(numpy.ma.array([22080], dtype=numpy.int16, fill_value=1),
        ...             clear_data=clear, not_clear_data=not_clear,
        ...             nodata=1)
        masked_array(data=[22080],
                     mask=[False],
               fill_value=1,
                    dtype=int16)

    Args:
        data (numpy.ma.masked_array): The QA Raster Data
        clear_data (List[float]): The bits values to be considered as Clear. Default is [].
        not_clear_data (List[float]): The bits values to be considered as Not Clear Values (Cloud,Shadow, etc).
        nodata (float): Pixel nodata value.
        confidence (QAConfidence): The confidence rules mapping. See more in :class:`~cube_builder.utils.image.QAConfidence`.

    Returns:
        numpy.ma.masked_array: An array which the values represent ``clear_data`` and the masked values represents ``not_clear_data``.
    """
    is_numpy_or_masked_array = type(data) in (numpy.ndarray, numpy.ma.masked_array)
    if type(data) in (float, int,):
        data = numpy.ma.masked_array([data])
    elif (isinstance(data, Iterable) and not is_numpy_or_masked_array) or (
            isinstance(data, numpy.ndarray) and not hasattr(data, 'mask')):
        data = numpy.ma.masked_array(data, mask=data == nodata, fill_value=nodata)
    elif not is_numpy_or_masked_array:
        raise TypeError(f'Expected a number or numpy masked array for {data}')

    if nodata is not None:
        data[data == nodata] = numpy.ma.masked

    # Cloud Confidence only once
    if confidence:
        data = confidence.apply(data)

    # Mask all not clear data before get any valid data
    for value in not_clear_data:
        if value == 2 and confidence:
            not_oli_positions = numpy.where(numpy.invert(confidence.oli))
            tmp_result = numpy.ma.masked_where(data.mask, data)
            tmp_result.mask[not_oli_positions] = True

            masked = extract_qa_bits(tmp_result, value)
            if isinstance(masked.mask, numpy.bool_):
                masked.mask = numpy.full(data.shape, fill_value=masked.mask, dtype=numpy.bool_)

            masked.mask[not_oli_positions] = data[not_oli_positions].mask

            tmp_result = None
        else:
            masked = extract_qa_bits(data, value)

        data = numpy.ma.masked_where(masked > 0, data)

    clear_mask = data.mask.copy()
    for value in clear_data:
        masked = numpy.ma.getdata(extract_qa_bits(data, value))
        clear_mask = numpy.ma.logical_or(masked > 0, clear_mask)

    if len(data.mask.shape) > 0:
        data = numpy.ma.masked_where(numpy.invert(clear_mask), data)
    else:  # Adapt to work with single value
        if clear_mask[0]:
            data.mask = numpy.invert(clear_mask)

    return data

