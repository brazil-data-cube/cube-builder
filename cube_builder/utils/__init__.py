#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
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
