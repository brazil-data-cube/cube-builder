#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Represent the data cube parameters to be attached in execution step."""

import sqlalchemy as sa
from bdc_catalog.models import Collection
from bdc_catalog.models.base_sql import BaseModel
from sqlalchemy.orm import relationship


class CubeParameters(BaseModel):
    """Define a table to store the data cube parameters to be passed during the execution."""

    __tablename__ = 'cube_parameters'

    id = sa.Column(sa.Integer, primary_key=True)
    collection_id = sa.Column(sa.ForeignKey(Collection.id, onupdate='CASCADE', ondelete='CASCADE'),
                              nullable=False)
    metadata_ = sa.Column(sa.JSON, default='{}', nullable=False)

    cube = relationship(Collection, lazy='select')

    __table_args__ = (
        sa.Index(None, collection_id),
        dict(schema='cube_builder'),
    )
