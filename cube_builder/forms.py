#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder forms used to validate both data input and data serialization."""

from bdc_catalog.models import Collection, GridRefSys, db
from marshmallow.fields import Float, Integer, String
from marshmallow_sqlalchemy.schema import ModelSchema


class CollectionForm(ModelSchema):
    """Form definition for Model Collection."""

    class Meta:
        """Internal meta information of Form interface."""

        model = Collection
        sqla_session = db.session
        exclude = ('extent', )


class GridRefSysForm(ModelSchema):
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

        model = GridRefSys
        sqla_session = db.session
        exclude = ('table_id', )
