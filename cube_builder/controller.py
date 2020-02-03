#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# 3rdparty
from flask import request
from flask_restplus import Namespace, Resource
from werkzeug.exceptions import BadRequest
from bdc_core.decorators.auth import require_oauth_scopes

# BDC Scripts
from .business import CubeBusiness
from .parsers import DataCubeParser, DataCubeProcessParser


api = Namespace('datastorm', description='datastorm')


@api.route('/create')
class CubeCreateController(Resource):
    @require_oauth_scopes(scope="bdc_scripts:datastorm:POST")
    def post(self):
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
    @require_oauth_scopes(scope="bdc_scripts:datastorm:POST")
    def post(self):
        args = request.get_json()

        form = DataCubeProcessParser()

        data = form.load(args)

        proc = CubeBusiness.maestro(data['datacube'], data['collections'], data['tiles'], data['start_date'], data['end_date'])

        return proc
