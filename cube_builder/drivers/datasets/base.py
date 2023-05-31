#
# This file is part of Cube Builder.
# Copyright (C) 2023 INPE.
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

"""Define the base interface for DataSet."""

from typing import Any, Dict, List, Optional, Union

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

OptionalDict = Optional[Dict[str, Any]]


class DataSet:
    """Represent the DataSet Reader abstraction for Rasterio/GDAL Drivers.

    This driver basically is used for generic datasets types.

    Examples:
        Open a simple GeoTIFF

        .. doctest::
           :skipif: True
            >>> from cube_builder.drivers.datasets import DataSet
            >>> dataset = DataSet("/path/to/my-scene.tif")
            >>> dataset.open()  # or with dataset:
            >>> arr2d = dataset.read(1, masked=True)
    """

    uri: str
    """The URI entry for dataset."""
    mode: str = 'r'
    """Dataset mode."""
    options: Dict[str, Any] = None
    """Optional options used to open a dataset."""
    dataset: Any = None
    """The Rasterio/GDAL dataset reference."""

    def __init__(self, uri: str, mode: str = 'r', extra_data=None, **options):
        """Build a basic dataset with rasterio support."""
        self.uri = uri
        self.mode = mode
        self.options = options
        self.dataset = None
        self.extra_data = extra_data or {}

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
        return f'DataSet(path={self.uri}, mode={self.mode})'
