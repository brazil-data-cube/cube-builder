#
# This file is part of Cube Builder.
# Copyright (C) 2023 INPE.
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

"""Define the base interface to deal with generic datasets that implements Zip format through VSICURL/VSIZIP."""

from .base import DataSet


class ZippedDataSet(DataSet):
    """Implement a DataSet that supports Zip compression."""

    def __init__(self, uri: str, mode: str = 'r', extra_data=None, **options):
        """Build a new dataset from ZIP files."""
        super().__init__(uri, mode, extra_data=extra_data, **options)
        self._setup()

    def _setup(self):
        flag = '/vsizip'
        if self.uri.startswith('http'):
            flag = f'{flag}//vsicurl'

        self._flag = flag
