#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder forms used to validate both data input and data serialization."""

from bdc_db.models import Band, Collection, TemporalCompositionSchema, db
from marshmallow.fields import Method, String
from marshmallow_sqlalchemy.schema import ModelSchema

from .models import Activity


class CollectionForm(ModelSchema):
    """Define Form definition for Model Collection."""

    class Meta:
        """Internal meta information of Form interface."""

        model = Collection
        sqla_session = db.session


class ActivityForm(ModelSchema):
    """Define Form definition for Model Activity."""

    class Meta:
        """Internal meta information of Form interface."""

        model = Activity
        sqla_session = db.session


class BandForm(ModelSchema):
    """Define form definition for model Band.

    Used to serialize band values.
    """

    class Meta:
        """Internal meta information of Form interface."""

        model = Band
        sqla_session = db.session


DEFAULT_TEMPORAL_UNITS = ['day', 'month']


class TemporalSchemaForm(ModelSchema):
    """Define form definition for model TemporalCompositionSchema."""

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
