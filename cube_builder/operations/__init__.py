#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Python Module which provides the well-known operations, such Median, Mean, etc in Cube Builder."""

from typing import Dict, Type

from .base import BaseOperation
from .median import Median


class Factory:
    functions: Dict[str, Type[BaseOperation]] = dict()

    def initialize(self):
        self.functions[Median.composite_function] = Median

    def get_operation(self, composite_function: str) -> Type[BaseOperation]:
        assert composite_function in self.functions.keys()

        return self.functions[composite_function]


factory = Factory()


__all__ = (
    'BaseOperation',
    'Median',
    'factory',
)
