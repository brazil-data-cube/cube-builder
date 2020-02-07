#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# 3rdparty
from sqlalchemy import Column, String, ARRAY, ForeignKey, Integer, JSON
from sqlalchemy.orm import relationship
from bdc_db.models.base_sql import BaseModel
from bdc_db.models import Collection


class Activity(BaseModel):
    __tablename__ = 'activities'
    __table_args__ = {"schema": "cube_builder"}

    id = Column(Integer, primary_key=True)
    collection_id = Column(String(64), nullable=False)
    warped_collection_id = Column(String(64), nullable=False)
    activity_type = Column('activity_type', String(64), nullable=False)
    period = Column(String(64), nullable=False)
    status = Column(String(64), nullable=False)
    args = Column('args', JSON)
    tags = Column('tags', ARRAY(String))
    scene_type = Column('scene_type', String)
    band = Column('band', String(64), nullable=False)

    # Relations
    # history = relationship('ActivityHistory', order_by='desc(ActivityHistory.start)')
