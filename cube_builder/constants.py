#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
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

FMASK_CLEAR_DATA = (0, 1)
"""Define the Clear Data values for the cloud masking Fmask4."""
FMASK_NOT_CLEAR_DATA = (2, 4)
"""Define the Not Clear Data values for the cloud masking Fmask4."""
