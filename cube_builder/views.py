#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Brazil Data Cube Cube Builder routes."""

from bdc_auth_client.decorators import oauth2
# 3rdparty
from flask import Blueprint, jsonify, request

# Cube Builder
from .celery.utils import list_queues
from .config import Config
from .controller import CubeController
from .forms import (CubeItemsForm, CubeStatusForm, DataCubeForm, DataCubeMetadataForm, DataCubeProcessForm,
                    GridRefSysForm, PeriodForm)
from .version import __version__

bp = Blueprint('cubes', import_name=__name__)


@bp.route('/', methods=['GET'])
def status():
    """Define a simple route to retrieve Cube-Builder API status."""
    return dict(
        message='Running',
        description='Cube Builder',
        version=__version__
    ), 200


@bp.route('/cube-status', methods=('GET', ))
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def cube_status(**kwargs):
    """Retrieve the cube processing state, which refers to total items and total to be done."""
    form = CubeStatusForm()

    args = request.args.to_dict()

    errors = form.validate(args)

    if errors:
        return errors, 400

    return jsonify(CubeController.get_cube_status(**args))


@bp.route('/cubes', defaults=dict(cube_id=None), methods=['GET'])
@bp.route('/cubes/<cube_id>', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_cubes(cube_id, **kwargs):
    """List all data cubes available."""
    if cube_id is not None:
        message, status_code = CubeController.get_cube(cube_id)

    else:
        message, status_code = CubeController.list_cubes()

    return jsonify(message), status_code


@bp.route('/cubes', methods=['POST'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["write"])
def create_cube(**kwargs):
    """Define POST handler for datacube creation.

    Expects a JSON that matches with ``DataCubeForm``.
    """
    form = DataCubeForm()

    args = request.get_json()

    errors = form.validate(args)

    if errors:
        return errors, 400

    data = form.load(args)

    cubes, status = CubeController.create(data)

    return jsonify(cubes), status

@bp.route('/cubes/<cube_id>', methods=['PUT'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["write"])
def update_cube_matadata(cube_id, **kwargs):
    """Define PUT handler for datacube Updation.

    Expects a JSON that matches with ``DataCubeMetadataForm``.
    """
    form = DataCubeMetadataForm()

    args = request.get_json()

    errors = form.validate(args)

    if errors:
        return errors, 400

    data = form.load(args)

    message, status = CubeController.update(cube_id, data)

    return jsonify(message), status


@bp.route('/cubes/<cube_id>/tiles', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_tiles(cube_id, **kwargs):
    """List all data cube tiles already done."""
    message, status_code = CubeController.list_tiles_cube(cube_id, only_ids=True)

    return jsonify(message), status_code


@bp.route('/cubes/<cube_id>/tiles/geom', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_tiles_as_features(cube_id, **kwargs):
    """List all tiles as GeoJSON feature."""
    message, status_code = CubeController.list_tiles_cube(cube_id)

    return jsonify(message), status_code


@bp.route('/cubes/<cube_id>/items', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_cube_items(cube_id, **kwargs):
    """List all data cube items."""
    form = CubeItemsForm()

    args = request.args.to_dict()

    errors = form.validate(args)

    if errors:
        return errors, 400

    message, status_code = CubeController.list_cube_items(cube_id, **args)

    return jsonify(message), status_code


@bp.route('/cubes/<cube_id>/meta', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def get_cube_meta(cube_id, **kwargs):
    """Retrieve the meta information of a data cube such STAC provider used, collection, etc."""
    message, status_code = CubeController.cube_meta(cube_id)

    return jsonify(message), status_code


@bp.route('/start', methods=['POST'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["write"])
def start_cube(**kwargs):
    """Define POST handler for datacube execution.

    Expects a JSON that matches with ``DataCubeProcessForm``.
    """
    args = request.get_json()

    form = DataCubeProcessForm()

    errors = form.validate(args)

    if errors:
        return errors, 400

    data = form.load(args)

    proc = CubeController.maestro(**data)

    return proc


@bp.route('/list-merges', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_merges(**kwargs):
    """Define POST handler for datacube execution.

    Expects a JSON that matches with ``DataCubeProcessForm``.
    """
    args = request.args

    res = CubeController.check_for_invalid_merges(**args)

    return res


@bp.route('/grids', defaults=dict(grs_id=None), methods=['GET'])
@bp.route('/grids/<grs_id>', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_grs_schemas(grs_id, **kwargs):
    """List all data cube Grids."""
    if grs_id is not None:
        result, status_code = CubeController.get_grs_schema(grs_id)
    else:
        result, status_code = CubeController.list_grs_schemas()

    return jsonify(result), status_code


@bp.route('/create-grids', methods=['POST'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["write"])
def create_grs(**kwargs):
    """Create the grid reference system using HTTP Post method."""
    form = GridRefSysForm()

    args = request.get_json()

    errors = form.validate(args)

    if errors:
        return errors, 400

    cubes, status = CubeController.create_grs_schema(**args)

    return cubes, status


@bp.route('/list-periods', methods=['POST'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_periods(**kwargs):
    """List data cube periods.

    The user must provide the following query-string parameters:
    - schema: Temporal Schema
    - step: Temporal Step
    - start_date: Start offset
    - last_date: End date offset
    """
    parser = PeriodForm()

    args = request.get_json()

    errors = parser.validate(args)

    if errors:
        return errors, 400

    return CubeController.generate_periods(**args)


@bp.route('/composite-functions', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_composite_functions(**kwargs):
    """List all data cube supported composite functions."""
    message, status_code = CubeController.list_composite_functions()

    return jsonify(message), status_code


@bp.route('/tasks', methods=['GET'])
@oauth2(required=Config.BDC_AUTH_REQUIRED, roles=["read"])
def list_tasks(**kwargs):
    """List all pending and running tasks on celery."""
    queues = list_queues()
    return queues
