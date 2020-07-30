#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Cube Builder Task Activity to track celery execution."""

from datetime import datetime
from typing import Union

# 3rdparty
from sqlalchemy import Column, Date, String, ARRAY, Integer, JSON, Text
from sqlalchemy.engine.result import ResultProxy
from bdc_db.models.base_sql import BaseModel, db
from bdc_db.models import Collection

from cube_builder.utils import get_cube_id


class Activity(BaseModel):
    """Define a SQLAlchemy model to track celery execution."""

    __tablename__ = 'activities'
    __table_args__ = {"schema": "cube_builder"}

    id = Column(Integer, primary_key=True)
    collection_id = Column(String(64), nullable=False)
    warped_collection_id = Column(String(64), nullable=False)
    activity_type = Column('activity_type', String(64), nullable=False)
    period = Column(String(64), nullable=False)
    date = Column(Date, nullable=False)
    tile_id = Column(String, nullable=False)
    status = Column(String(64), nullable=False)
    args = Column('args', JSON)
    tags = Column('tags', ARRAY(String))
    scene_type = Column('scene_type', String)
    band = Column('band', String(64), nullable=False)
    traceback = Column(Text(), nullable=True)

    @classmethod
    def list_merge_files(cls, collection: Collection, tile: str,
                         start_date: Union[str, datetime],
                         end_date: Union[str, datetime]) -> ResultProxy:
        """List all merge files used in data cube generation.

        Notes:
            In order to seek all images used, we seek for the collection IDENTITY.

        TODO: Convert this function to SQLAlchemy expression instead raw SQL.

        Returns:
            Retrieves a ResultProxy with merges found.
        """
        collection_expression = 'warped_collection_id'
        collection_id = get_cube_id(collection.id)

        sql = """
        SELECT id, tile_id, band, date::VARCHAR as date, collection_id, args->'dataset'::VARCHAR AS data_set, (elem->>'link')::VARCHAR as link, status, traceback::TEXT, args->'file' AS file
          FROM cube_builder.activities
         CROSS JOIN json_array_elements(args->'assets') elem
         WHERE {} = '{}'
           AND tile_id = '{}'
           AND date BETWEEN '{}'::DATE AND '{}'::DATE
         ORDER BY id
        """.format(
            collection_expression,
            collection_id,
            tile,
            start_date,
            end_date
        )

        res = db.session.execute(sql)

        return res.fetchall()
