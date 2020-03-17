#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Brazil Data Cube Scripts Blueprint strategy."""

from flask import Blueprint
from flask_restplus import Api

from .controller import api as cube_ns

bp = Blueprint('cubes', __name__, url_prefix='/api')

api = Api(bp, doc=False)

api.add_namespace(cube_ns)
