#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Brazil Data Cube Cube Builder routes."""

# 3rdparty
from flask import current_app, request, jsonify

# Cube Builder
from .business import CubeBusiness
from .forms import GrsSchemaForm, RasterSchemaForm, TemporalSchemaForm
from .parsers import DataCubeParser, DataCubeProcessParser, PeriodParser, CubeStatusParser, ListMergeParser, \
    ListCubeItemParser

from .version import __version__


@current_app.route("/", methods=["GET"])
def status():
    """Retrieve application status health check."""
    return dict(
        message='Running',
        description='Cube Builder',
        version=__version__
    )


@current_app.route('/create', methods=('POST', ))
def create_cube():
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

    return dict(message='Cube created', cubes=cubes), status


@current_app.route('/start', methods=('POST', ))
def start_cube():
    """Define POST handler for datacube execution.

    Expects a JSON that matches with ``DataCubeProcessParser``.
    """
    args = request.get_json()

    form = DataCubeProcessParser()

    data = form.load(args)

    proc = CubeBusiness.maestro(**data)

    return jsonify(proc)


@current_app.route('/list-merges')
def list_merges():
    """Define POST handler for datacube execution.

    Expects a JSON that matches with ``DataCubeProcessParser``.
    """
    args = request.args

    errors = ListMergeParser().validate(args)

    if errors:
        return jsonify(errors), 400

    res = CubeBusiness.check_for_invalid_merges(**args)

    return res


@current_app.route('/create-temporal-composition', methods=('POST', ))
def create_temporal_composition():
    """Create the temporal composite schema using HTTP Post method.

    Expects a JSON that matches with ``TemporalSchemaParser``.
    """
    form = TemporalSchemaForm()

    args = request.get_json()

    errors = form.validate(args)

    if errors:
        return errors, 400

    cubes, status = CubeBusiness.create_temporal_composition(args)

    return jsonify(cubes), status


@current_app.route('/create-raster-size', methods=('POST', ))
def create_raster_size():
    """Create the raster schema using HTTP Post method."""
    form = RasterSchemaForm()

    args = request.get_json()

    errors = form.validate(args)

    if errors:
        return errors, 400

    cubes, status = CubeBusiness.create_raster_schema(**args)

    return cubes, status


@current_app.route('/create-grs', methods=('POST', ))
def create_grs():
    """Create the raster schema using HTTP Post method."""
    form = GrsSchemaForm()

    args = request.get_json()

    errors = form.validate(args)

    if errors:
        return errors, 400

    grs, status = CubeBusiness.create_grs_schema(**args)

    return jsonify(grs), status


@current_app.route('/timeline', methods=('GET', ))
def list_periods():
    """List data cube periods.

    The user must provide the following query-string parameters:
    - schema: Temporal Schema
    - step: Temporal Step
    - start_date: Start offset
    - last_date: End date offset
    """
    parser = PeriodParser()

    args = request.args

    errors = parser.validate(args)

    if errors:
        return errors, 400

    return jsonify(CubeBusiness.generate_periods(**args))


@current_app.route('/raster-size', methods=('GET', ))
def list_raster_size():
    raster_sizes = CubeBusiness.list_raster_size()

    return jsonify(raster_sizes), 200


@current_app.route('/cube-status', methods=('GET', ))
def cube_status():
    form = CubeStatusParser()

    args = request.args.to_dict()

    errors = form.validate(args)

    if errors:
        return errors, 400

    return CubeBusiness.get_cube_status(**args)


@current_app.route('/cubes', defaults=dict(cube_id=None), methods=['GET'])
@current_app.route('/cubes/<cube_id>', methods=['GET'])
def list_cubes(cube_id):
    if cube_id is not None:
        response = CubeBusiness.get_cube(cube_id)
    else:
        response = CubeBusiness.list_cubes()

    return jsonify(response), 200


@current_app.route('/cubes/<cube_id>/tiles', methods=['GET'])
def list_tiles_as_features(cube_id):
    features = CubeBusiness.list_tiles_cube(cube_id)

    return jsonify(features), 200


@current_app.route('/grs', defaults=dict(grs_id=None), methods=['GET'])
@current_app.route('/grs/<grs_id>', methods=['GET'])
def list_grs_schemas(grs_id):
    if grs_id is not None:
        response = CubeBusiness.get_grs_schema(grs_id)
    else:
        response = CubeBusiness.list_grs_schemas()

    return jsonify(response), 200


@current_app.route('/temporal-composition', methods=['GET'])
def list_temporal_composition():
    response = CubeBusiness.list_temporal_composition()

    return jsonify(response)


@current_app.route('/composite-functions', methods=['GET'])
def list_composite_functions():
    response = CubeBusiness.list_composite_functions()

    return jsonify(response)


@current_app.route('/cubes/<cube_id>/items/tiles', methods=['GET'])
def list_items_tiles(cube_id):
    tiles = CubeBusiness.list_cube_items_tiles(cube_id)

    return jsonify(tiles)


@current_app.route('/cubes/<cube_id>/meta', methods=['GET'])
def get_cube_meta(cube_id: str):
    """Retrieve the meta information of a data cube such STAC provider used, collection, etc."""
    message = CubeBusiness.get_cube_meta(cube_id)

    return jsonify(message)


@current_app.route('/cubes/<cube_id>/items', methods=['GET'])
def list_cube_items(cube_id):
    args = request.args.to_dict()

    form = ListCubeItemParser()

    errors = form.validate(args)

    if errors:
        return jsonify(errors), 400

    message = CubeBusiness.list_cube_items(cube_id, **args)

    return jsonify(message), 200
