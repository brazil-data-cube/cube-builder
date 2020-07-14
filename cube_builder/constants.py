#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define the Brazil Data Cube Builder constants."""


CLEAR_OBSERVATION_NAME = 'ClearOb'

CLEAR_OBSERVATION_ATTRIBUTES = dict(
    name=CLEAR_OBSERVATION_NAME,
    description='Clear Observation Count.',
    data_type='uint8',
    min=0,
    max=255,
    fill=0,
    scale=1,
    common_name=CLEAR_OBSERVATION_NAME,
)

TOTAL_OBSERVATION_NAME = 'TotalOb'

TOTAL_OBSERVATION_ATTRIBUTES = dict(
    name=TOTAL_OBSERVATION_NAME,
    description='Total Observation Count',
    data_type='uint8',
    min=0,
    max=255,
    fill=0,
    scale=1,
    common_name=TOTAL_OBSERVATION_NAME,
)

PROVENANCE_NAME = 'Provenance'

PROVENANCE_ATTRIBUTES = dict(
    name=PROVENANCE_NAME,
    description='Provenance value Day of Year',
    data_type='int16',
    min=1,
    max=366,
    fill=-1,
    scale=1,
    common_name=PROVENANCE_NAME,
)
