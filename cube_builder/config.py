#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""Brazil Data Cube Configuration."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def get_settings(env):
    """Retrieve Config class from environment."""
    return CONFIG.get(env)


class Config:
    """Base configuration with default flags."""

    CELERYD_PREFETCH_MULTIPLIER = 1  # disable
    DEBUG = False
    TESTING = False
    DATA_DIR = '/data'
    ACTIVITIES_SCHEMA = 'cube_builder'
    RABBIT_MQ_URL = os.environ.get('RABBIT_MQ_URL', 'pyamqp://guest@localhost')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'SQLALCHEMY_DATABASE_URI',
        'postgresql://postgres:bdc-scripts2019@localhost:5435/bdc'
    )
    STAC_URL = os.environ.get(
        'STAC_URL',
        'http://brazildatacube.dpi.inpe.br/bdc-stac/0.7.0/'
    )
    SECRET_KEY = 'cube-builder'
    TASK_RETRY_DELAY = int(os.environ.get('TASK_RETRY_DELAY', 60 * 60))  # a hour


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
    "DevelopmentConfig": DevelopmentConfig(),
    "ProductionConfig": ProductionConfig(),
    "TestingConfig": TestingConfig()
}