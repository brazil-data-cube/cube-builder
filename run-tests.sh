#!/usr/bin/env bash
#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

pydocstyle cube_builder tests setup.py && \
isort cube_builder tests setup.py --check-only --diff -l 120 && \
check-manifest --ignore ".drone.yml,.readthedocs.*" && \
sphinx-build -qnW --color -b doctest docs/sphinx/ docs/sphinx/_build/doctest && \
pytest
