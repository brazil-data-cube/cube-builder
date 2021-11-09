#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""Brazil Data Cube Configuration."""

import os
from distutils.util import strtobool

from .version import __version__

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def get_settings(env=None):
    """Retrieve Config class from environment."""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')

    return CONFIG.get(env)


class Config:
    """Base configuration with default flags."""

    DEBUG = False
    TESTING = False

    # Factor to reserve tasks of the broker. By default, multiply by 1
    CELERYD_PREFETCH_MULTIPLIER = 1 * int(os.environ.get('CELERYD_PREFETCH_MULTIPLIER', 1))
    CBERS_AUTH_TOKEN = os.environ.get('CBERS_AUTH_TOKEN', '')
    # Path to store data
    ACTIVITIES_SCHEMA = 'cube_builder'
    DATA_DIR = os.environ.get('DATA_DIR', '/data')
    WORK_DIR = os.environ.get('WORK_DIR', '/workdir')
    RABBIT_MQ_URL = os.environ.get('RABBIT_MQ_URL', 'pyamqp://guest:guest@localhost')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'SQLALCHEMY_DATABASE_URI',
        'postgresql://postgres:postgres@localhost:5432/bdc_catalog'
    )
    STAC_URL = os.environ.get('STAC_URL', 'https://brazildatacube.dpi.inpe.br/stac/')
    MAX_THREADS_IMAGE_VALIDATOR = int(os.environ.get('MAX_THREADS_IMAGE_VALIDATOR', os.cpu_count()))
    # rasterio
    RASTERIO_ENV = dict(
        GDAL_DISABLE_READDIR_ON_OPEN=True,
    )
    # Access Token
    FLASK_ACCESS_TOKEN = os.getenv('FLASK_ACCESS_TOKEN', None)

    # Add prefix path to the items. Base path is /Repository
    # So the asset will be /Repository/Mosaic/collectionName/version/tile/period/scene.tif
    ITEM_PREFIX = os.getenv('ITEM_PREFIX', '/cubes')

    # BDC-Auth OAuth2
    BDC_AUTH_CLIENT_ID = os.getenv('BDC_AUTH_CLIENT_ID', None)
    BDC_AUTH_CLIENT_SECRET = os.getenv('BDC_AUTH_CLIENT_SECRET', None)
    BDC_AUTH_ACCESS_TOKEN_URL = os.getenv('BDC_AUTH_ACCESS_TOKEN_URL', None)
    BDC_AUTH_REQUIRED = strtobool(os.getenv('BDC_AUTH_REQUIRED', '0'))

    # CBERS URL Prefix
    CBERS_SOURCE_URL_PREFIX = os.getenv('CBERS_SOURCE_URL_PREFIX', 'cdsr.dpi.inpe.br/api/download/TIFF')
    CBERS_TARGET_URL_PREFIX = os.getenv('CBERS_TARGET_URL_PREFIX', 'www.dpi.inpe.br/catalog/tmp')

    REDOC = {'title': 'Cube Builder API Doc', 'version': __version__}

    QUEUE_IDENTITY_CUBE = os.getenv('QUEUE_IDENTITY_CUBE', 'merge-cube')
    QUEUE_PREPARE_CUBE = os.getenv('QUEUE_PREPARE_CUBE', 'prepare-cube')
    QUEUE_BLEND_CUBE = os.getenv('QUEUE_BLEND_CUBE', 'blend-cube')
    QUEUE_PUBLISH_CUBE = os.getenv('QUEUE_PUBLISH_CUBE', 'publish-cube')


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


CONFIG = {
    "development": DevelopmentConfig(),
    "production": ProductionConfig(),
    "testing": TestingConfig()
}
