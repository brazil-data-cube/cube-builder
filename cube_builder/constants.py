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

"""Define the Brazil Data Cube Builder constants."""


CLEAR_OBSERVATION_NAME = 'CLEAROB'

CLEAR_OBSERVATION_ATTRIBUTES = dict(
    name=CLEAR_OBSERVATION_NAME,
    description='Clear Observation Count.',
    data_type='uint8',
    min_value=0,
    max_value=255,
    nodata=0,
    scale=1,
    common_name='ClearOb',
)

TOTAL_OBSERVATION_NAME = 'TOTALOB'

TOTAL_OBSERVATION_ATTRIBUTES = dict(
    name=TOTAL_OBSERVATION_NAME,
    description='Total Observation Count',
    data_type='uint8',
    min_value=0,
    max_value=255,
    nodata=0,
    scale=1,
    common_name='TotalOb',
)

PROVENANCE_NAME = 'PROVENANCE'

PROVENANCE_ATTRIBUTES = dict(
    name=PROVENANCE_NAME,
    description='Provenance value Day of Year',
    data_type='int16',
    min_value=1,
    max_value=366,
    nodata=-1,
    scale=1,
    common_name='Provenance',
)

# Band for Combined Collections
DATASOURCE_NAME = 'DATASOURCE'

DATASOURCE_ATTRIBUTES = dict(
    name=DATASOURCE_NAME,
    description='Data set value',
    data_type='uint8',
    min_value=0,
    max_value=254,
    nodata=255,
    scale=1,
    common_name='datasource',
)

COG_MIME_TYPE = 'image/tiff; application=geotiff; profile=cloud-optimized'

PNG_MIME_TYPE = 'image/png'

SRID_ALBERS_EQUAL_AREA = 100001


def to_bool(val: str):
    """Convert a string representation to true or false.

    This method was adapted from `pypa/distutils <https://github.com/pypa/distutils>`_
    to avoid import deprecated module.

    The following values are supported:
    - ``True``: 'y', 'yes', 't', 'true', 'on', and '1'
    - ``False``: 'n', 'no', 'f', 'false', 'off', and '0'

    Raises:
        ValueError: When the given string value could not be converted to boolean.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1',):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0',):
        return 0

    raise ValueError(f"invalid boolean value for {val}")
