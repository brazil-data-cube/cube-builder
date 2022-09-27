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

import urllib.parse
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


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
        self.uri = uri
        self.mode = mode
        self.options = options
        self.dataset = None

    def open(self, *args, **kwargs):
        """Open a dataset using rasterio/gdal way."""
        self.dataset = rasterio.open(self.uri, mode=self.mode, **self.options)

    @property
    def meta(self) -> OptionalDict:
        """The dataset metadata.

        Note:
            Make sure to call ``DataSet.open()``.
        """
        return self._get_prop('meta')

    @property
    def profile(self) -> OptionalDict:
        """The dataset raster profile.

        Note:
            Make sure to call ``DataSet.open()``.
        """
        return self._get_prop('profile')

    @property
    def transform(self) -> Optional[Affine]:
        """The dataset GeoTransform.

        Note:
            Make sure to call ``DataSet.open()``.
        """
        return self._get_prop('transform')

    @property
    def crs(self) -> Optional[CRS]:
        """The dataset coordinate reference system.

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
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        return f'DataSet(path={self.uri})'


class ZippedDataSet(DataSet):
    """Implement a DataSet that supports Zip compression."""

    def __init__(self, uri: str, mode: str = 'r', **options):
        super().__init__(uri, mode, **options)
        self._setup()

    def _setup(self):
        flag = '/vsizip'
        if self.uri.startswith('http'):
            flag = '/vsicurl'

        self._flag = flag


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
        self.band = band
        self._uri = uri
        super().__init__(uri, mode, **options)

    def open(self, band: str = None, *args, **kwargs):
        ds = rasterio.open(f'{self._flag}/{self.uri}')
        if len(ds.subdatasets) != 4:
            raise RuntimeError(f'Invalid data set Sentinel {self.uri}')

        band = band or self.band
        # TODO: Complete validation
        if band in ('B02', 'B03', 'B04', 'B08'):
            target = ds.subdatasets[0]
        elif band in ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12'):
            target = ds.subdatasets[1]
        elif band in ('B01', 'B09', 'B10'):
            target = ds.subdatasets[2]
        elif band == 'TCI':
            target = ds.subdatasets[3]
        else:
            raise ValueError(f'Invalid band for Sentinel-2 DataSet {band}')
        self.uri = target
        super().open(*args, **kwargs)

    def close(self):
        super().close()
        self.uri = self._uri


def dataset_from_uri(uri: str, band: str = None, **options) -> DataSet:
    """Build a Cube Builder dataset from a string URI.

    Args:
        uri: Path to the dataset. Any URI supported by GDAL dataset.
        band: Internal band name for dataset.
        **options: The rasterio/gdal dataset options.
    """
    parse = urlparse(uri)

    fragments = parse.path.split(':')
    path = fragments[0]

    if parse.scheme and parse.scheme.startswith('zip+') or path.endswith('.zip'):
        # Zip or SentinelZip
        return _make_zip_dataset(parse, band=band, **options)

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
