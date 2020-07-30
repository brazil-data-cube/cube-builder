#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define the interface for Median data cubes operation."""

import numpy
from .base import BaseOperation, StackMaskedArray


class Median(BaseOperation):
    composite_function = 'MED'

    def calc(self, stack_masked: StackMaskedArray) -> StackMaskedArray:
        return numpy.ma.median(stack_masked, axis=0).data
