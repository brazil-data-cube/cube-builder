#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Python Module for Cube Builder."""

from json import JSONEncoder

from bdc_catalog.ext import BDCCatalog
from flask import Flask, abort, request
from werkzeug.exceptions import HTTPException, InternalServerError

from . import celery, config
from .version import __version__


def create_app(config_name=None):
    """Create Brazil Data Cube application from config object.

    Args:
        config_name (string) Config instance name
    Returns:
        Flask Application with config instance scope
    """
    app = Flask(__name__)

    conf = config.get_settings(config_name)

    app.config.from_object(conf)

    with app.app_context():
        # Initialize Flask SQLAlchemy
        BDCCatalog(app)

        # Just make sure to initialize db before celery
        celery_app = celery.create_celery_app(app)
        celery.celery_app = celery_app

        setup_app(app)

    return app


def setup_error_handlers(app: Flask):
    """Configure Cube Builder Error Handlers on Flask Application."""
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Handle exceptions."""
        if isinstance(e, HTTPException):
            return {'code': e.code, 'description': e.description}, e.code

        app.logger.exception(e)

        return {'code': InternalServerError.code,
                'description': InternalServerError.description}, InternalServerError.code


def setup_app(app):
    """Configure internal middleware for Flask app."""

    @app.after_request
    def after_request(response):
        """Enable CORS."""
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        response.headers.add('Access-Control-Allow-Headers',
                             'Origin, X-Requested-With, Content-Type, Accept, Authorization, X-Api-Key')
        return response

    @app.before_request
    def middleware():
        """Intercept requests, validating access token if it configured."""
        if request.method != 'OPTIONS':
            if app.config.get('FLASK_ACCESS_TOKEN') is not None:
                if request.headers.get('X-Api-Key') != app.config.get('FLASK_ACCESS_TOKEN'):
                    abort(403, 'Forbidden')

    class ImprovedJSONEncoder(JSONEncoder):
        def default(self, o):
            from datetime import datetime
            
            if isinstance(o, set):
                return list(o)
            if isinstance(o, datetime):
                return o.isoformat()
            return super(ImprovedJSONEncoder, self).default(o)

    app.json_encoder = ImprovedJSONEncoder

    setup_error_handlers(app)

    from .views import bp
    app.register_blueprint(bp)
