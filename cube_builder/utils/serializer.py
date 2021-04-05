# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""File responsible for data serialization."""

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
            result[c.key] = value
        return result

    @staticmethod
    def serialize_list(l):
        """Serialize a list of SQLAlchemy objects."""
        return [m.serialize() for m in l]
