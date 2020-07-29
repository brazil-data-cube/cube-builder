#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder forms used to validate both data input and data serialization."""

from bdc_db.models import (Band, Collection, GrsSchema, RasterSizeSchema,
                           TemporalCompositionSchema, db, CompositeFunctionSchema, CollectionItem)
from marshmallow.fields import Float, Integer, Method, String
from marshmallow_sqlalchemy.schema import ModelSchema

from .models import Activity


class CollectionForm(ModelSchema):
    """Form definition for Model Collection."""

    class Meta:
        """Internal meta information of Form interface."""

        model = Collection
        sqla_session = db.session


class ActivityForm(ModelSchema):
    """Form definition for Model Activity."""

    class Meta:
        """Internal meta information of Form interface."""

        model = Activity
        sqla_session = db.session


class BandForm(ModelSchema):
    """Form definition for model Band.

    Used to serialize band values.
    """

    class Meta:
        """Internal meta information of Form interface."""

        model = Band
        sqla_session = db.session


DEFAULT_TEMPORAL_UNITS = ['day', 'month']


class TemporalSchemaForm(ModelSchema):
    """Form definition for model TemporalCompositionSchema."""

    class Meta:
        """Internal meta information of Form interface."""

        model = TemporalCompositionSchema
        sqla_session = db.session

    id = String(dump_only=True)
    temporal_composite_unit = Method(deserialize='load_temporal_composite')

    def load_temporal_composite(self, value):
        """Validate temporal composite with all supported values."""
        if value in DEFAULT_TEMPORAL_UNITS:
            return value

        raise RuntimeError('Invalid temporal unit. Supported values: {}'.format(DEFAULT_TEMPORAL_UNITS))


class RasterSchemaForm(ModelSchema):
    """Form definition for the model RasterSizeSchema."""

    id = String(dump_only=True)
    grs_schema = String(required=True, load_only=True)
    resolution = Integer(required=True, load_only=True)
    raster_size_x = Integer(required=True, dump_only=True)
    raster_size_y = Integer(required=True, dump_only=True)
    chunk_size_x = Integer(required=True, load_only=True)
    chunk_size_y = Integer(required=True, load_only=True)

    class Meta:
        """Internal meta information of form interface."""

        model = RasterSizeSchema
        sqla_session = db.session


class GrsSchemaForm(ModelSchema):
    """Form definition for the model GrsSchema."""

    id = String(dump_only=True)
    name = String(required=True, load_only=True)
    projection = String(required=True, load_only=True)
    meridian = Integer(required=True, load_only=True)
    degreesx = Float(required=True, load_only=True)
    degreesy = Float(required=True, load_only=True)
    bbox = String(required=True, load_only=True)

    class Meta:
        """Internal meta information of form interface."""

        model = GrsSchema
        sqla_session = db.session


class CompositeFunctionForm(ModelSchema):
    class Meta:
        model = CompositeFunctionSchema
        sqla_session = db.session


class CollectionItemForm(ModelSchema):
    tile_id = String(dump_only=True)

    class Meta:
        model = CollectionItem
        sqla_session = db.session
