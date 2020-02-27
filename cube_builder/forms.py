#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder forms used to validate both data input and data serialization."""

from bdc_db.models import db, Collection
from marshmallow_sqlalchemy.schema import ModelSchema

from cube_builder.models import Activity


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
