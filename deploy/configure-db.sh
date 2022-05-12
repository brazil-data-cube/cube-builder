#!/bin/bash

#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

bdc-db db init # Create database and schema
bdc-db db create-namespaces # Schemas
bdc-db db create-extension-postgis # PostGIS
lccs-db db create-extension-hstore # PostGIS
bdc-db db create-schema # Up tables
bdc-db db create-triggers
cube-builder load-data