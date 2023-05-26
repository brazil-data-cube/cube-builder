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

"""File responsible for data serialization."""

from datetime import datetime

from decimal import Decimal

from sqlalchemy.inspection import inspect


class Serializer(object):
    """Class responsible to serialize database SQLAlchemy instances."""

    @staticmethod
    def serialize(obj):
        """Serialize a SQLAlchemy instance."""
        result = dict()
        for c in inspect(obj).mapper.column_attrs:
            value = getattr(obj, c.key)
            if type(value) == Decimal:
                value = float(value)
            elif isinstance(value, datetime):
                # Ensure RFC339 is used
                value = value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            result[c.key] = value
        return result

    @staticmethod
    def serialize_list(l):
        """Serialize a list of SQLAlchemy objects."""
        return [m.serialize() for m in l]
