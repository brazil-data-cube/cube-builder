#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder parsers."""

from marshmallow import Schema, fields


class DataCubeBandParser(Schema):
    """Parse data cube input bands."""

    names = fields.List(fields.String, required=True, allow_none=False)
    min = fields.Integer(required=True, allow_none=False)
    max = fields.Integer(required=True, allow_none=False)
    fill = fields.Integer(required=True, allow_none=False)
    scale = fields.Integer(required=True, allow_none=False)
    data_type = fields.String(required=True, allow_none=False)


class DataCubeParser(Schema):
    """Define parser for datacube creation."""

    datacube = fields.String(required=True, allow_none=False)
    grs = fields.String(required=True, allow_none=False)
    resolution = fields.Integer(required=True, allow_none=False)
    temporal_schema = fields.String(required=True, allow_none=False)
    bands_quicklook = fields.List(fields.String, required=True, allow_none=False)
    composite_function_list = fields.List(fields.String, required=True, allow_none=False)
    bands = fields.List(fields.String, required=True, allow_none=False)
    description = fields.String(required=True, allow_none=False)


class DataCubeProcessParser(Schema):
    """Define parser for datacube generation."""

    datacube = fields.String(required=True, allow_none=False)
    collections = fields.List(fields.String, required=True, allow_none=False)
    tiles = fields.List(fields.String, required=True, allow_none=False)
    start_date = fields.Date()
    end_date = fields.Date()
    bands = fields.List(fields.String, required=False)
    force = fields.Boolean(required=False, default=False)
