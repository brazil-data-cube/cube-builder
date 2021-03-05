#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder forms used to validate both data input and data serialization."""

from bdc_catalog.models import Collection, GridRefSys, db
from marshmallow import Schema, fields, pre_load
from marshmallow.validate import OneOf, Regexp, ValidationError
from marshmallow_sqlalchemy.schema import ModelSchema
from rasterio.dtypes import dtype_ranges


class CollectionForm(ModelSchema):
    """Form definition for Model Collection."""

    class Meta:
        """Internal meta information of Form interface."""

        model = Collection
        sqla_session = db.session
        exclude = ('extent', )


class GridRefSysForm(ModelSchema):
    """Form definition for the model GrsSchema."""

    id = fields.String(dump_only=True)
    name = fields.String(required=True, load_only=True)
    projection = fields.String(required=True, load_only=True)
    meridian = fields.Integer(required=True, load_only=True)
    degreesx = fields.Float(required=True, load_only=True)
    degreesy = fields.Float(required=True, load_only=True)
    bbox = fields.String(required=True, load_only=True)

    class Meta:
        """Internal meta information of form interface."""

        model = GridRefSys
        sqla_session = db.session
        exclude = ('table_id', )


INVALID_CUBE_NAME = 'Invalid data cube name. Expected only letters and numbers.'
SUPPORTED_DATA_TYPES = list(dtype_ranges.keys())


class BandDefinition(Schema):
    """Define a simple marshmallow structure for data cube bands on creation."""

    name = fields.String(required=True, allow_none=False)
    common_name = fields.String(required=True, allow_none=False)
    data_type = fields.String(required=True, allow_none=False, validate=OneOf(SUPPORTED_DATA_TYPES))
    metadata = fields.Dict(required=False, allow_none=False)


class DataCubeForm(Schema):
    """Define parser for datacube creation."""

    datacube = fields.String(required=True, allow_none=False, validate=Regexp('^[a-zA-Z0-9-]*$', error=INVALID_CUBE_NAME))
    grs = fields.String(required=True, allow_none=False)
    resolution = fields.Integer(required=True, allow_none=False)
    temporal_composition = fields.Dict(required=True, allow_none=False)
    bands_quicklook = fields.List(fields.String, required=True, allow_none=False)
    composite_function = fields.String(required=True, allow_none=False)
    bands = fields.Nested(BandDefinition, required=True, allow_none=False, many=True)
    quality_band = fields.String(required=True, allow_none=False)
    indexes = fields.Nested(BandDefinition, many=True)
    metadata = fields.Dict(required=True, allow_none=True)
    description = fields.String(required=True, allow_none=False)
    version = fields.Integer(required=True, allow_none=False, default=1)
    title = fields.String(required=True, allow_none=False)
    # Set cubes as public by default.
    public = fields.Boolean(required=False, allow_none=False, default=True)
    # Is Data cube generated from Combined Collections?
    is_combined = fields.Boolean(required=False, allow_none=False, default=False)

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
            import json
            import pkgutil

            import bdc_catalog
            from jsonschema import draft7_format_checker, validate
            content = pkgutil.get_data(bdc_catalog.__name__, 'jsonschemas/collection-temporal-composition-schema.json')
            schema = json.loads(content)
            try:
                schema['$id'] = schema['$id'].replace('#', '')
                validate(instance=data['temporal_schema'], schema=schema, format_checker=draft7_format_checker)
            except Exception as e:
                print(e)
                raise

        return data


class DataCubeMetadataForm(Schema):
    """Define parser for datacube updation."""

    metadata = fields.Dict(required=False, allow_none=True)
    description = fields.String(required=False, allow_none=False)
    title = fields.String(required=False, allow_none=False)
    public = fields.Boolean(required=False, allow_none=False, default=True)


class DataCubeProcessForm(Schema):
    """Define parser for datacube generation."""

    datacube = fields.String(required=True, allow_none=False)
    collections = fields.List(fields.String, required=True, allow_none=False)
    tiles = fields.List(fields.String, required=True, allow_none=False)
    start_date = fields.Date()
    end_date = fields.Date()
    bands = fields.List(fields.String, required=False)
    force = fields.Boolean(required=False, default=False)
    with_rgb = fields.Boolean(required=False, default=False)
    token = fields.String(required=False, allow_none=True)
    stac_url = fields.String(required=False, allow_none=True)
    shape = fields.List(fields.Integer(required=False))
    # Reuse data cube from another data cube
    reuse_from = fields.String(required=False, allow_none=True)
    histogram_matching = fields.Boolean(required=False, default=False)
    mask = fields.Dict()


class PeriodForm(Schema):
    """Define parser for Data Cube Periods."""

    schema = fields.String(required=True, allow_none=False)
    step = fields.Integer(required=True)
    unit = fields.String(required=True)
    start_date = fields.String(required=False)
    last_date = fields.String(required=False)
    cycle = fields.Dict(required=False, allow_none=True)
    intervals = fields.List(fields.String, required=False, allow_none=True)


class CubeStatusForm(Schema):
    """Parser for access data cube status resource."""

    cube_name = fields.String(required=True, allow_none=False)


class CubeItemsForm(Schema):
    """Parser for access data cube items resource."""

    tiles = fields.String(required=False)
    bbox = fields.String(required=False)
    start = fields.String(required=False)
    end = fields.String(required=False)
    page = fields.Integer(required=False)
    per_page = fields.Integer(required=False)