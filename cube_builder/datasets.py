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

"""Define the DataSet readers for Cube Builder."""
import os.path
import tarfile
import urllib.parse
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import Affine

from cube_builder.config import Config

OptionalDict = Optional[Dict[str, Any]]


class DataSet:
    """Represent the DataSet Reader abstraction for Rasterio/GDAL Drivers."""

    uri: str
    """The URI entry for dataset."""
    mode: str = 'r'
    """Dataset mode."""
    options: Dict[str, Any] = None
    """Optional options used to open a dataset."""
    dataset: Any = None
    """The Rasterio/GDAL dataset reference."""

    def __init__(self, uri: str, mode: str = 'r', **options):
        """Build a basic dataset with rasterio support."""
        self.uri = uri
        self.mode = mode
        self.options = options
        self.dataset = None

    def open(self, *args, **kwargs):
        """Open a dataset using rasterio/gdal way."""
        self.dataset = rasterio.open(self.uri, mode=self.mode, **self.options)

    @property
    def meta(self) -> OptionalDict:
        """Retrieve the dataset metadata.

        Note:
            Make sure to call ``DataSet.open()``.
        """
        return self._get_prop('meta')

    @property
    def profile(self) -> OptionalDict:
        """Retrieve the dataset raster profile.

        Note:
            Make sure to call ``DataSet.open()``.
        """
        return self._get_prop('profile')

    @property
    def transform(self) -> Optional[Affine]:
        """Retrieve the dataset GeoTransform.

        Note:
            Make sure to call ``DataSet.open()``.
        """
        return self._get_prop('transform')

    @property
    def crs(self) -> Optional[CRS]:
        """Retrieve the dataset coordinate reference system.

        Note:
            Make sure to call ``DataSet.open()``.
        """
        return self._get_prop('crs')

    def _get_prop(self, prop: str) -> Any:
        return getattr(self.dataset, prop, None)

    def close(self):
        """Close the current dataset."""
        if self.dataset:
            self.dataset.close()
            self.dataset = None

    def __enter__(self):
        """Add support to open dataset using Python Enter contexts."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close dataset automatically when out of context."""
        return self.close()

    def read(self, indexes: Union[int, List[int]], *args, **kwargs):
        """Implement the data set reader with rasterio/GDAL.

        Args:
            indexes: The int band value or list of band indices to read.

        KeywordArgs:
            window: Read data using rasterio window limits.
            masked: Flag to read data using
                `Numpy Masked Module <https://numpy.org/doc/stable/reference/maskedarray.generic.html#the-numpy-ma-module>`_ where nodata is masked.
        """
        return self.dataset.read(indexes, *args, **kwargs)

    def __repr__(self):
        """Represent a basic dataset as string."""
        return f'DataSet(path={self.uri})'


class ZippedDataSet(DataSet):
    """Implement a DataSet that supports Zip compression."""

    def __init__(self, uri: str, mode: str = 'r', **options):
        """Build a new dataset from ZIP files."""
        super().__init__(uri, mode, **options)
        self._setup()

    def _setup(self):
        flag = '/vsizip'
        if self.uri.startswith('http'):
            flag = f'{flag}//vsicurl'

        self._flag = flag


class LandsatTgzDataSet(ZippedDataSet):
    """Implement a DataSet that supports read LANDSAT tar gz datasets."""

    SUPPORTED_BAND_NAMES = [f'B{idx}' for idx in range(1, 8)] + ['QA_PIXEL', 'QA_RADSAT', '']

    def __init__(self, uri: str, scene_id: str, mode: str = 'r', band: str = None, **options):
        """Create a new Landsat compressed dataset."""
        self.band = band
        self.scene_id = scene_id
        self._uri = uri
        super().__init__(uri, mode, **options)

    def open(self, band: str = None, *args, **kwargs):
        """Try to open a Landsat compressed file as dataset."""
        band = band or self.band
        if band.startswith('SR_') or band.startswith('ST') or band in self.SUPPORTED_BAND_NAMES:
            suffix = '.TIF'
        else:
            raise ValueError(f'Invalid band for Sentinel-2 DataSet {band}')
        self.uri = f'{self._uri}/{self.scene_id}_{band}{suffix}'


class SentinelZipDataSet(ZippedDataSet):
    """Implement a DataSet that supports read Sentinel-2 datasets inside a generic Zip.

    Example:
        >>> from cube_builder.datasets import SentinelZipDataSet
        >>> ds = SentinelZipDataSet('S2A_MSIL2A_20220621T133201_N0400_R081_T22LGL_20220621T182715.zip')
        >>> with ds.open(band='B01'):  # Move internal points to the band B01 60m coastal.
        >>>     arr = ds.read(1)  # You must pass index here

    TODO: Implement way to specify multiple bands inside a Zip and open dinamically
        ds.open(band='B02,B04,B8A')
        arr3d = ds.read((1,2,3) # Read B02,B04,B8A, respectively
    """

    def __init__(self, uri: str, mode: str = 'r', band: str = None, **options):
        """Create a new Sentinel-2 dataset."""
        self.band = band
        self._uri = uri
        self._uri_parsed = urlparse(uri)
        self._band_descriptions = None
        super().__init__(uri, mode, **options)

    def open(self, band: str = None, *args, **kwargs):
        """Open Sentinel-2 subdataset inside zip file."""
        if self._uri_parsed.scheme in ('http', 'https',):
            self.uri = self.uri.split('?')[0]

        ds = rasterio.open(f'{self._flag}/{self.uri}')
        if len(ds.subdatasets) != 4:
            raise RuntimeError(f'Invalid data set Sentinel {self.uri}')

        band = band or self.band
        # TODO: Complete validation
        if band in ('B02', 'B03', 'B04', 'B08'):
            target = ds.subdatasets[0]  # 4 bands
        elif band in ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'):
            target = ds.subdatasets[1]  # 9 bands
        elif band in ('B01', 'B09', 'B10'):
            target = ds.subdatasets[2]
        elif band == 'TCI':
            target = ds.subdatasets[3]
        else:
            raise ValueError(f'Invalid band for Sentinel-2 DataSet {band}')
        self.uri = target
        super().open(*args, **kwargs)
        self._band_descriptions = self.dataset.descriptions

    def close(self):
        """Close a dataset."""
        super().close()
        self.uri = self._uri

    def read(self, indexes: Union[int, List[int]], *args, **kwargs):
        """Implement the data set reader with rasterio/GDAL.

        Args:
            indexes: The int band value or list of band indices to read.

        KeywordArgs:
            window: Read data using rasterio window limits.
            masked: Flag to read data using
                `Numpy Masked Module <https://numpy.org/doc/stable/reference/maskedarray.generic.html#the-numpy-ma-module>`_ where nodata is masked.
        """
        native_mode = kwargs.pop('native_mode', False)
        if self._band_descriptions and not native_mode:
            band_lst = [_resolve_sentinel_band_name(entry.split(',')[0]) for entry in self._band_descriptions]
            indexes = band_lst.index(self.band) + 1
        return self.dataset.read(indexes, *args, **kwargs)


def _resolve_sentinel_band_name(name: str) -> str:
    band_id = name[1:]
    if band_id.isnumeric():
        band_id = '{0:02d}'.format(int(band_id))
    return f'{name[0]}{band_id}'


def dataset_from_uri(uri: str, band: str = None, **options) -> DataSet:
    """Build a Cube Builder dataset from a string URI.

    Args:
        uri: Path to the dataset. Any URI supported by GDAL dataset.
        band: Internal band name for dataset.
        **options: The rasterio/gdal dataset options.
    """
    parse = urlparse(uri)

    path = parse.path.rsplit('?')[0]

    if parse.scheme and parse.scheme.startswith('zip+') or path.endswith('.zip'):
        # Zip or SentinelZip
        return _make_zip_dataset(parse, band=band, **options)
    elif parse.scheme and parse.scheme.startswith('vsitar+') or (path.endswith('.tar') or path.endswith('.tar.gz')):
        return _make_tgz_dataset(parse, band=band, **options)

    return DataSet(uri, **options)


def _make_zip_dataset(parse: urllib.parse.ParseResultBytes, band: str = None, **options) -> ZippedDataSet:
    fragments = parse.path.rsplit(':')
    path = fragments[0]
    band_ = band
    if band is None and len(fragments) > 1:
        band_ = fragments[1]
    if band_:
        return SentinelZipDataSet(parse._replace(path=path).geturl(), band=band_, **options)
    return ZippedDataSet(parse.geturl(), **options)


def _make_tgz_dataset(parse: urllib.parse.ParseResultBytes, band: str = None, **options) -> DataSet:
    # When vsicurl, download first 512 bytes
    url = parse.geturl()
    url_ = url
    for fragment in ['/vsicurl/', '/vsitar/', 'vsitar/', '/vsitar']:
        url_ = url_.replace(fragment, '')
    if '/vsicurl/' in url or url.startswith('http'): # TODO: Add FTP? or url.startswith('ftp://'):
        req = requests.get(url_, headers={'Range': 'bytes=0-511'}, verify=Config.RASTERIO_ENV['GDAL_HTTP_UNSAFESSL'],
                           timeout=30)
        if req.status_code > 206:
            raise RuntimeError(f'Could not detect Tar GZ Dataset {url}')
        info = tarfile.TarInfo.frombuf(req.content, tarfile.ENCODING, '')
    else:
        url_ = url
        for fragment in ['/vsicurl/', '/vsitar/']:
            url_ = url_.replace(fragment, '')

        with tarfile.open(url_) as fd:
            files = fd.getnames()
            if len(files) == 0:
                raise IOError(f'Invalid file {url}: No files found.')
            info = files[0]

    name = os.path.splitext(info.name)[0]
    if is_landsat_like(name):
        # Remove Band from scene id
        name = '_'.join(name.split('_')[:-1])
        return LandsatTgzDataSet(parse.geturl(), band=band, scene_id=name, **options)

    return ZippedDataSet(url, band=band, **options)


def is_landsat_like(name: str) -> bool:
    """Identify if the given values is a Landsat program scene id."""
    fragments = name.split('_')
    if len(fragments) != 8:
        return False
    sensor = fragments[0]
    satellite = sensor[-2:]
    return satellite.isnumeric() and int(satellite) in (4, 5, 7, 8, 9)
