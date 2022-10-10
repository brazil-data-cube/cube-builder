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

"""Module to deal with Local directory data accessor.

This module aims to load files from disk and use to generate datacubes.
It was intended to work following `GDALCubes Formats <https://github.com/appelmar/gdalcubes>`_ and
then provides a minimal interface to seek files in DISK using regular expression.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, TypedDict

import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform


class BandDef(TypedDict):
    """Define a minimal struct to represent Band definition."""

    nodata: float
    scale: float
    pattern: str


BandMap = Dict[str, BandDef]


class DataFormat(dict):
    """Represent a JSON Data Format for data seeking.

    A :class:`cube_builder.local_accessor.DataFormat` represents a minimal pattern interface to
    seek data in disk. It was designed to adopt the `GDALCubes Formats <https://github.com/appelmar/gdalcubes>`_.
    """

    def __init__(self, name: str, bands: BandMap, pattern: str, images: str, datetime: dict, **kwargs):
        """Build a new Local Data directory seeker."""
        required_values = dict(
            bands=bands,
            pattern=pattern,
            images=images,
            datetime=datetime,
            name=name
        )
        super().__init__(**required_values, **kwargs)

    @property
    def name(self):
        """Retrieve the data format unique name."""
        return self['name']

    @property
    def description(self):
        """Retrieve the detailed description for DataFormat."""
        return self['description']

    @property
    def bands(self) -> BandMap:
        """Retrieve the band mapping for data seeking in disk."""
        return self['bands']


class ImageCollection:
    """Represent a structure to deal with directory listing.

    Use the method :meth:`cube_builder.local_accessor.ImageCollection.files` to list all
    files from a directory:

    Examples:
        You may consider download other formats
        from `GDALCubes Formats <https://github.com/appelmar/gdalcubes/tree/master/formats>`_.

        >>> from cube_builder.local_accessor import load_format
        >>> data_accessor = load_format('examples/formats/Sentinel2_L2A.json')
        >>> files = data_accessor.files('/data/input-dir/sentinel-2', recursive=True, pattern='.tif')
        >>> print(files)  # The list of files found using ``examples/formats/Sentinel2_L2A.json``.

    Once the files are found, you can restrict the found data intersecting with a `Region of Interest`(``ROI``)
    as following:

    Examples:
        >>> from shapely.geometry import box
        >>> roi = box(-54, -12, -53.9, -11.9)
        >>> data_accessor = load_format('examples/formats/Sentinel2_L2A.json')
        >>> files = data_accessor.files('/data/input-dir/sentinel-2', recursive=True, pattern='.tif')
        >>> data = data_accessor.create_collection(roi, '<ROI_Proj4>', files=files)
        >>> print(data)  # The list of files found using ``examples/formats/Sentinel2_L2A.json`` that intersects with ROI.

    Warning:
        Yoy may have performance issues in complex directory structure or
        the directory has several folders/files in disk,
    """

    _format: DataFormat

    def __init__(self, fmt: DataFormat):
        """Build a new image collection data handler."""
        self._format = fmt

    def files(self, folder: str, pattern: str = None, recursive: bool = False) -> List[Path]:
        """Apply directory ``glob`` and then retrieve the matched files."""
        path = Path(folder)
        attribute = 'rglob' if recursive else 'glob'
        pattern = pattern or self._format['pattern']

        out = []

        for entry in getattr(path, attribute)(f'*/*'):
            match = re.search(pattern, entry.name)
            if match:
                out.append(entry)

        return out

    def create_collection(self, tile: Polygon, tile_proj4: str, files: List[Path],
                          start: datetime, end: datetime) -> dict:
        """Create a Python dictionary containing the matched files given temporal spatial extent restriction.

        Args:
            tile (Polygon): A Shapely geometry to filter
            tile_proj4 (str): Shapely Geometry CRS proj4
            files (List[Path]): List of files to iterate and apply spatial filter.
            start (datetime): Start date filter.
            end (datetime): End date filter.
        """
        tile_crs = CRS.from_proj4(tile_proj4)
        out = {}
        for file in files:
            found = None
            for band, ctx in self._format.bands.items():
                if band in file.name and re.search(ctx['pattern'], file.name):
                    found = band
                    break

            if found is None:
                continue

            nodata = self._format.bands[found].get('nodata', None)

            # Match date
            match_datetime = re.search(self._format['datetime']['pattern'], str(file))
            if not match_datetime:
                continue

            datetime_fmt = self._format['datetime']['format']
            image_date = datetime.strptime(match_datetime.groups()[0], datetime_fmt)
            if not (start <= image_date <= end):
                continue

            key = image_date.strftime('%Y-%m-%d')
            out.setdefault(found, {})
            out[found].setdefault(key, {})
            out[found][key].setdefault(self._format.name, [])

            with rasterio.open(file) as ds:
                bbox = box(*ds.bounds)
                crs = CRS.from_proj4(ds.crs.to_proj4())
                transformer = Transformer.from_crs(crs, tile_crs, always_xy=True).transform

                image_geom = transform(transformer, bbox)
                if image_geom.intersects(tile):
                    scene = {
                        "link": str(file),
                        "nodata": nodata,
                        "dataset": self._format.name
                    }

                    out[found][key][self._format.name].append(scene)

        return out


def load_format(file_path: str) -> ImageCollection:
    """Load a JSON file as :class:`cube_builder.local_accessor.DataFormat` and create ImageCollection.

    Notes:
        The argument ``file_path`` must exist, otherwise a ``IOError`` is raised.

    Args:
        file_path (str): Path to the JSON file format.

    Returns:
        ImageCollection: A prepared Image Collection that seeks for files.
    """
    with open(file_path) as fd:
        data = json.load(fd)
    data['name'] = Path(file_path).stem

    return ImageCollection(DataFormat(**data))
