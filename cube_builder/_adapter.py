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

"""Define basic module to adapt Python libraries like STAC v1 and legacy versions."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import requests
import shapely.geometry
from pystac_client import Client
from stac import STAC as _STAC


class BaseSTAC(ABC):
    """Define base class to represent a STAC interface to communicate with Server."""

    uri: str
    """Represent URI for server."""
    headers: dict
    """Represent HTTP headers to be attached in requests."""
    params: dict
    """Represent HTTP parameters for requests."""

    def __init__(self, uri: str, params=None, headers=None, **kwargs):
        """Build STAC signature."""
        self.uri = uri
        self.params = params
        self.headers = headers
        self._options = kwargs

    @abstractmethod
    def search(self, **parameters) -> dict:
        """Search for collection items on STAC server."""

    @abstractmethod
    def items(self, collection_id: str, **kwargs) -> dict:
        """Access STAC Collection Items."""

    @abstractmethod
    def collections(self) -> List[dict]:
        """Retrieve the collections from STAC."""

    @abstractmethod
    def collection(self, collection_id: str) -> dict:
        """Access STAC Collection."""

    @staticmethod
    def _items_result(features: List[dict], matched: int):
        return {
            "context": {
                "returned": len(features),
                "matched": matched
            },
            "features": features
        }


class STACV1(BaseSTAC):
    """Define structure to add support for STAC v1.0+.

    This implementation uses `pystac-client <https://pystac-client.readthedocs.io/en/latest/>`_
    to communicate with STAC v1.0.
    """

    def __init__(self, uri: str, params=None, headers=None, **kwargs):
        """Build STAC instance."""
        super(STACV1, self).__init__(uri, params, headers, **kwargs)

        self._instance = Client.open(uri, headers=headers, parameters=params, **kwargs)

    def search(self, limit=10, max_items=10, **parameters) -> dict:
        """Search for collection items on STAC server."""
        max_items = limit
        item_search = self._instance.search(limit=limit, max_items=max_items, **parameters)

        items = item_search.items()
        items = [i.to_dict() for i in items]

        return self._items_result(items, matched=item_search.matched())

    def collections(self) -> List[dict]:
        """Retrieve the collections from STAC."""
        return [c.to_dict() for c in self._instance.get_collections()]

    def collection(self, collection_id: str) -> dict:
        """Access STAC Collection."""
        collection = self._instance.get_collection(collection_id)
        return collection.to_dict()

    def items(self, collection_id: str, **kwargs) -> dict:
        """Access STAC Collection Items."""
        collection = self._instance.get_collection(collection_id)

        items = collection.get_items()
        items = [i.to_dict() for i in items]

        result = self.search(collections=[collection_id], limit=1, max_items=1)

        return self._items_result(items, matched=result['context']['matched'])


class STACLegacy(BaseSTAC):
    """Define structure to add support for legacy versions of STAC server..

    This implementation uses `stac.py <https://github.com/brazil-data-cube/stac.py>`_
    to communicate with STAC legacy versions 0.8x, 0.9x.
    """

    def __init__(self, uri: str, params=None, headers=None, **kwargs):
        """Build STAC instance."""
        super(STACLegacy, self).__init__(uri, params, headers, **kwargs)

        params = params or {}
        headers = headers or {}
        token = params.get('access_token') or (headers.get('x-api-key') or headers.get('X-Api-Key'))

        self._instance = _STAC(uri, access_token=token)

    def search(self, **parameters) -> dict:
        """Search for collection items on STAC server."""
        options = deepcopy(parameters)
        # Remove unsupported values
        options.pop('query', None)
        try:
            return self._instance.search(filter=options)
        except:
            # Use bbox instead
            geom = options.pop('intersects', None)
            if geom is None:
                raise

            options['bbox'] = shapely.geometry.shape(geom).bounds

            return self._instance.search(filter=options)

    def collections(self) -> List[dict]:
        """Retrieve the collections from STAC."""
        collections = self._instance.collections
        return list(collections.values())

    def collection(self, collection_id: str) -> dict:
        """Access STAC Collection."""
        return self._instance.collection(collection_id)

    def items(self, collection_id: str, **kwargs) -> dict:
        """Access STAC Collection Items."""
        return self.search(collections=[collection_id], limit=1)


def build_stac(uri, headers=None, **parameters) -> BaseSTAC:
    """Build a STAC instance according versions."""
    response = requests.get(uri, headers=headers, params=parameters)

    response.raise_for_status()

    catalog = response.json()
    if not catalog.get('stac_version'):
        raise RuntimeError(f'Invalid STAC "{uri}", missing "stac_version"')

    stac_version = catalog['stac_version']
    if stac_version.startswith('0.'):
        return STACLegacy(uri, params=parameters, headers=headers)
    return STACV1(uri, params=parameters, headers=headers)
