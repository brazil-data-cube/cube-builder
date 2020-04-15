#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Brazil Data Cube Cube Builder routes."""

# 3rdparty
from flask import request
from flask_restplus import Namespace, Resource
from werkzeug.exceptions import BadRequest

# BDC Scripts
from .business import CubeBusiness
from .forms import TemporalSchemaForm
from .parsers import DataCubeParser, DataCubeProcessParser

api = Namespace('cubes', description='cubes')


@api.route('/create')
class CubeCreateController(Resource):
    """Define route for datacube creation."""

    def post(self):
        """Define POST handler for datacube creation.

        Expects a JSON that matches with ``DataCubeParser``.
        """
        form = DataCubeParser()

        args = request.get_json()

        errors = form.validate(args)

        if errors:
            return errors, 400

        data = form.load(args)

        cubes, status = CubeBusiness.create(data)

        return cubes, status


@api.route('/process')
class CubeProcessController(Resource):
    """Define route for datacube execution."""

    def post(self):
        """Define POST handler for datacube execution.

        Expects a JSON that matches with ``DataCubeProcessParser``.
        """
        args = request.get_json()

        form = DataCubeProcessParser()

        data = form.load(args)

        proc = CubeBusiness.maestro(**data)

        return proc


@api.route('/list-merges')
class CubeMergeStatusController(Resource):
    """Define route for datacube execution."""

    def get(self):
        """Define POST handler for datacube execution.

        Expects a JSON that matches with ``DataCubeProcessParser``.
        """
        args = request.args

        res = CubeBusiness.check_for_invalid_merges(**args)

        return res


@api.route('/create-temporal-schema')
class TemporalSchemaController(Resource):
    """Define route for TemporalCompositeSchema creation."""

    def post(self):
        """Create the temporal composite schema using HTTP Post method.

        Expects a JSON that matches with ``TemporalSchemaParser``.
        """
        form = TemporalSchemaForm()

        args = request.get_json()

        errors = form.validate(args)

        if errors:
            return errors, 400

        cubes, status = CubeBusiness.create_temporal_composition(args)

        return cubes, status
