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

"""Define the module to support Landsat Collection 2 dataset (in tar format)."""

from ..compressed import ZippedDataSet


class LandsatTgzDataSet(ZippedDataSet):
    """Implement a DataSet that supports read LANDSAT tar gz datasets."""

    SUPPORTED_BAND_NAMES = [f'B{idx}' for idx in range(1, 8)] + ['QA_PIXEL', 'QA_RADSAT', '']

    def __init__(self, uri: str, scene_id: str, mode: str = 'r', band: str = None, **options):
        """Create a new Landsat compressed dataset."""
        self.band = band
        self.scene_id = scene_id
        self._uri = uri
        super().__init__(uri, mode, **options)

    def open(self, band: str = None, *args, **kwargs):
        """Try to open a Landsat compressed file as dataset."""
        band = band or self.band
        if band.startswith('SR_') or band.startswith('ST') or band in self.SUPPORTED_BAND_NAMES:
            suffix = '.TIF'
        else:
            raise ValueError(f'Invalid band for Sentinel-2 DataSet {band}')
        self.uri = f'{self._uri}/{self.scene_id}_{band}{suffix}'


def is_landsat_like(name: str) -> bool:
    """Identify if the given values is a Landsat program scene id."""
    fragments = name.split('_')
    if len(fragments) != 8:
        return False
    sensor = fragments[0]
    satellite = sensor[-2:]
    return satellite.isnumeric() and int(satellite) in (4, 5, 7, 8, 9)
