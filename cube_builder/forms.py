#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

from bdc_db.models import db, Collection
from marshmallow_sqlalchemy.schema import ModelSchema


class CollectionForm(ModelSchema):
    class Meta:
        model = Collection
        sqla_session = db.session