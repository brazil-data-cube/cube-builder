# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Cube Builder utilities."""

from typing import List

import geoalchemy2
import sqlalchemy.sql.expression
from geoalchemy2 import func
from sqlalchemy import Column


def get_srid_column(columns: List[Column], default_srid=4326) -> Column:
    """Retrieve a PostGIS SRID column from a list of defined column.

    Note:
        When there is no SRID column or Geometry SRID associated, use const value `default_srid`.
    """
    out = sqlalchemy.bindparam('srid', default_srid)
    for column in columns:
        if isinstance(column.type, geoalchemy2.Geometry) and column.type.srid != -1:
            out = func.ST_SRID(column)
        if _is_srid_column(column):
            out = column
    return out


def _is_srid_column(column: Column):
    """Identify the SQLAlchemy column and check if its srid and has relationship with PostGIS spatial_ref_sys.

    Note:
        Only PostGIS module supported.
    """
    if column.name == 'srid' and len(column.foreign_keys) > 0:
        for fk in column.foreign_keys:
            if 'spatial_ref_sys.srid' in fk.target_fullname:
                return True
    return False
