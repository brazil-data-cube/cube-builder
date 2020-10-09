from sqlalchemy.inspection import inspect
from decimal import Decimal

class Serializer(object):
    @staticmethod
    def serialize(obj):
        result = dict()
        for c in inspect(obj).mapper.column_attrs:
            value = getattr(obj, c.key)
            if type(value) == Decimal:
                value = float(value)
            result[c.key] = value
        return result

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]
