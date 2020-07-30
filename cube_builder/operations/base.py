#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define base interface for Cube Builder Operation."""

from pathlib import Path
from typing import Tuple

import numpy
from rasterio.windows import Window


StackMaskedArray = numpy.ma.MaskedArray


class BaseOperation:
    composite_function: str = None
    datacube: str = None

    def __init__(self, datacube: str, file_path: Path, shape: Tuple[int, int], profile, nodata):
        self.datacube = datacube

        self.nodata = nodata
        self.profile = profile
        self.file_path = file_path
        # TODO: Open SmartDataSet and write in disk blocks instead allocated entire raster in-memory
        self.raster = numpy.full(shape, fill_value=nodata, dtype=profile['dtype'])

    def __call__(self, stack_masked: StackMaskedArray, window: Window, nodata: int, notdonemask: numpy.ndarray):
        """Execute operation which applies the result into internal array.

        This functions requires the method `calc` to be implemented.
        You can customize and apply your own expression into stacked array, such median, mean, avg, etc.

        Args:
             stack_masked - Stacked array
        """
        array = self.calc(stack_masked)
        array[notdonemask.astype(numpy.bool_)] = nodata

        row_offset = window.row_off + window.height
        col_offset = window.col_off + window.width
        self.raster[window.row_off: row_offset, window.col_off: col_offset] = array.astype(self.profile['dtype'])

    def calc(self, stack_masked: StackMaskedArray) -> StackMaskedArray:
        """Perform the operation calc.

        Notes:
            The `stack_masked` is a 3d-array, which represents the stacked images and their x,y values.
        """
        raise NotImplementedError()

    def persist(self):
        """Save result raster in disk."""
        from ..utils import save_as_cog

        save_as_cog(str(self.file_path), self.raster, 'w', **self.profile)
