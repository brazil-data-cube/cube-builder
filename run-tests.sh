#!/usr/bin/env bash
#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

pydocstyle cube_builder && \
isort --check-only --diff --recursive cube_builder/*.py && \
check-manifest --ignore ".readthedocs.*" && \
sphinx-build -qnW --color -b doctest docs/sphinx/ docs/sphinx/_build/doctest # && \
pytest
