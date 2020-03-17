#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Brazil Data Cube main package executor."""

import os

from .cli import main

if __name__ == '__main__':
    main(as_module=True)
