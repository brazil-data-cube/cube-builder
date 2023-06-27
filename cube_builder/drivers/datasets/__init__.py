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

"""Define the module for custom dataset management."""

import os
import tarfile
from urllib.parse import ParseResultBytes, urlparse

import requests

from ...config import Config
from .base import DataSet
from .compressed import ZippedDataSet
from .landsat import LandsatTgzDataSet, is_landsat_like
from .sentinel import Sentinel2BaseLine, Sentinel2DataSet, SentinelZipDataSet


def dataset_from_uri(uri: str, band: str = None, extra_data=None, **options) -> DataSet:
    """Build a Cube Builder dataset from a string URI.

    Args:
        uri: Path to the dataset. Any URI supported by GDAL dataset.
        band: Internal band name for dataset.
        extra_data: Optional dict values used to create/open dataset.
        **options: The rasterio/gdal dataset options.
    """
    parse = urlparse(uri)

    path = parse.path.rsplit('?')[0]

    if parse.scheme and parse.scheme.startswith('zip+') or path.endswith('.zip'):
        # Zip or SentinelZip
        return _make_zip_dataset(parse, band=band, **options)
    elif parse.scheme and parse.scheme.startswith('vsitar+') or (path.endswith('.tar') or path.endswith('.tar.gz')):
        return _make_tgz_dataset(parse, band=band, **options)

    extra_data = extra_data or {}
    extra_data.setdefault("band", band)
    if extra_data.get("sceneid") and Sentinel2DataSet.is_sentinel_scene(extra_data["sceneid"]):
        return Sentinel2DataSet(uri, extra_data=extra_data, band=band, **options)

    return DataSet(uri, extra_data=None, **options)


def _make_zip_dataset(parse: ParseResultBytes, band: str = None, extra_data=None, **options) -> ZippedDataSet:
    fragments = parse.path.rsplit(':')
    path = fragments[0]
    band_ = band
    if band is None and len(fragments) > 1:
        band_ = fragments[1]
    if band_:
        return SentinelZipDataSet(parse._replace(path=path).geturl(), band=band_, **options)
    return ZippedDataSet(parse.geturl(), **options)


def _make_tgz_dataset(parse: ParseResultBytes, band: str = None, extra_data=None, **options) -> DataSet:
    # When vsicurl, download first 512 bytes
    url = parse.geturl()
    url_ = url
    for fragment in ['/vsicurl/', '/vsitar/', 'vsitar/', '/vsitar']:
        url_ = url_.replace(fragment, '')
    if '/vsicurl/' in url or url.startswith('http'): # TODO: Add FTP? or url.startswith('ftp://'):
        req = requests.get(url_, headers={'Range': 'bytes=0-511'}, verify=Config.RASTERIO_ENV['GDAL_HTTP_UNSAFESSL'],
                           timeout=30)
        if req.status_code > 206:
            raise RuntimeError(f'Could not detect Tar GZ Dataset {url}')
        info = tarfile.TarInfo.frombuf(req.content, tarfile.ENCODING, '')
    else:
        url_ = url
        for fragment in ['/vsicurl/', '/vsitar/']:
            url_ = url_.replace(fragment, '')

        with tarfile.open(url_) as fd:
            files = fd.getnames()
            if len(files) == 0:
                raise IOError(f'Invalid file {url}: No files found.')
            info = files[0]

    name = os.path.splitext(info.name)[0]
    if is_landsat_like(name):
        # Remove Band from scene id
        name = '_'.join(name.split('_')[:-1])
        return LandsatTgzDataSet(parse.geturl(), band=band, scene_id=name, **options)

    return ZippedDataSet(url, band=band, **options)

