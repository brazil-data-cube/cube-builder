#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder parsers."""

from marshmallow import Schema, fields, pre_load
from marshmallow.validate import OneOf, Regexp, ValidationError
from rasterio.dtypes import dtype_ranges


class DataCubeBandParser(Schema):
    """Parse data cube input bands."""

    names = fields.List(fields.String, required=True, allow_none=False)
    min = fields.Integer(required=True, allow_none=False)
    max = fields.Integer(required=True, allow_none=False)
    fill = fields.Integer(required=True, allow_none=False)
    scale = fields.Integer(required=True, allow_none=False)
    data_type = fields.String(required=True, allow_none=False)


INVALID_CUBE_NAME = 'Invalid data cube name. Expected only letters and numbers.'
SUPPORTED_DATA_TYPES = list(dtype_ranges.keys())


class BandDefinition(Schema):
    name = fields.String(required=True, allow_none=False)
    common_name = fields.String(required=True, allow_none=False)
    data_type = fields.String(required=True, allow_none=False, validate=OneOf(SUPPORTED_DATA_TYPES))
    metadata = fields.Dict(required=False, allow_none=False)


class DataCubeParser(Schema):
    """Define parser for datacube creation."""

    datacube = fields.String(required=True, allow_none=False, validate=Regexp('^[a-zA-Z0-9]*$', error=INVALID_CUBE_NAME))
    grs = fields.String(required=True, allow_none=False)
    resolution = fields.Integer(required=True, allow_none=False)
    temporal_schema = fields.Dict(required=True, allow_none=False)
    bands_quicklook = fields.List(fields.String, required=True, allow_none=False)
    composite_function = fields.String(required=True, allow_none=False, validate=OneOf(['MED', 'STK', 'IDT']))
    bands = fields.Nested(BandDefinition, required=True, allow_none=False, many=True)
    quality_band = fields.String(required=True, allow_none=False)
    indexes = fields.Nested(BandDefinition, many=True)
    description = fields.String(required=True, allow_none=False)
    license = fields.String(required=False, allow_none=True)
    version = fields.Integer(required=True, allow_none=False, default=1)
    title = fields.String(required=True, allow_none=False)
    # Set cubes as public by default.
    public = fields.Boolean(required=False, allow_none=False, default=True)

    @pre_load
    def validate_indexes(self, data, **kwargs):
        """Ensure that both indexes and quality band is present in attribute 'bands'.

        Seeks for quality_band in attribute 'bands' and set as `common_name`.

        Raises:
            ValidationError when a band inside indexes or quality_band is duplicated with attribute bands.
        """
        indexes = data['indexes']

        band_names = [b['name'] for b in data['bands']]

        for band_index in indexes:
            if band_index['name'] in band_names:
                raise ValidationError(f'Duplicated band name in indices {band_index["name"]}')

        if 'quality_band' in data:
            if data['quality_band'] not in band_names:
                raise ValidationError(f'Quality band "{data["quality_band"]}" not found in key "bands"')

            band = next(filter(lambda band: band['name'] == data['quality_band'], data['bands']))
            band['common_name'] = 'quality'

        if 'temporal_schema' in data:
            import pkgutil
            import bdc_catalog
            from jsonschema import validate, draft7_format_checker
            import json
            content = pkgutil.get_data(bdc_catalog.__name__, 'jsonschemas/collection-temporal-composition-schema.json')
            schema = json.loads(content)
            try:
                schema['$id'] = schema['$id'].replace('#', '')
                validate(instance=data['temporal_schema'], schema=schema, format_checker=draft7_format_checker)
            except Exception as e:
                print(e)
                raise

        return data


class DataCubeProcessParser(Schema):
    """Define parser for datacube generation."""

    datacube = fields.String(required=True, allow_none=False)
    collections = fields.List(fields.String, required=True, allow_none=False)
    tiles = fields.List(fields.String, required=True, allow_none=False)
    start_date = fields.Date()
    end_date = fields.Date()
    bands = fields.List(fields.String, required=False)
    force = fields.Boolean(required=False, default=False)
    with_rgb = fields.Boolean(required=False, default=False)
    token = fields.String(required=False, allow_none=False)


class PeriodParser(Schema):
    """Define parser for Data Cube Periods."""

    schema = fields.String(required=True, allow_none=False)
    step = fields.Integer(required=True)
    start_date = fields.String(required=False)
    last_date = fields.String(required=False)
