#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# 3rdparty
from celery.backends.database import Task
from sqlalchemy import Column, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from bdc_db.models.base_sql import BaseModel


class DataStormActivityHistory(BaseModel):
    __tablename__ = 'datastorm_activity_history'

    activity_id = Column(ForeignKey('datastorm_activities.id'), primary_key=True, nullable=False)
    task_id = Column(ForeignKey(Task.id), primary_key=True, nullable=False)
    start = Column('start', DateTime)
    env = Column('env', JSON)

    # Relations
    activity = relationship('DataStormActivity', back_populates="history")
    task = relationship(Task, uselist=False)
