#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""Brazil Data Cube Configuration."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def get_settings(env=None):
    """Retrieve Config class from environment."""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')

    return CONFIG.get(env)


class Config:
    """Base configuration with default flags."""

    # Factor to reserve tasks of the broker. By default, multiply by 1
    CELERYD_PREFETCH_MULTIPLIER = 1 * int(os.environ.get('CELERYD_PREFETCH_MULTIPLIER', 1))
    CBERS_AUTH_TOKEN = os.environ.get('CBERS_AUTH_TOKEN', '')
    DEBUG = False
    TESTING = False
    # Path to store data
    DATA_DIR = os.environ.get('DATA_DIR', '/data')
    ACTIVITIES_SCHEMA = 'cube_builder'
    RABBIT_MQ_URL = os.environ.get('RABBIT_MQ_URL', 'pyamqp://guest@localhost')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'SQLALCHEMY_DATABASE_URI',
        'postgresql://postgres:bdc-scripts2019@localhost:5435/bdc_scripts'
    )
    STAC_URL = os.environ.get(
        'STAC_URL',
        'http://brazildatacube.dpi.inpe.br/bdc-stac/0.8.1'
    )
    SECRET_KEY = 'cube-builder'
    MAX_THREADS_IMAGE_VALIDATOR = int(os.environ.get('MAX_THREADS_IMAGE_VALIDATOR', os.cpu_count()))
    # rasterio
    RASTERIO_ENV = dict(
        GDAL_DISABLE_READDIR_ON_OPEN=True,
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'
    )


class ProductionConfig(Config):
    """Production Mode."""

    DEBUG = False


class DevelopmentConfig(Config):
    """Development Mode."""

    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    """Testing Mode (Continous Integration)."""

    TESTING = True
    DEBUG = True


key = Config.SECRET_KEY

CONFIG = {
    "development": DevelopmentConfig(),
    "production": ProductionConfig(),
    "testing": TestingConfig()
}
