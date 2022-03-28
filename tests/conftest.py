#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Represent the minimal features for cube builder tests."""

import json
import os

import pytest
from pkg_resources import resource_filename

from cube_builder.celery.worker import app as _app
from cube_builder.celery.worker import celery


@pytest.fixture()
def app():
    """Create Flask Application and return a initialized app."""
    with _app.app_context():
        yield _app


@pytest.fixture()
def client(app):
    """Retrieve the Flask Test Client interface."""
    with app.test_client() as _client:
        yield _client


@pytest.fixture(scope="session")
def json_data():
    """Prepare tests fixture data."""
    mocks_dir = resource_filename(__name__, 'data/json')
    mocks_files = os.listdir(mocks_dir)
    mocks = dict()
    for filename in mocks_files:
        absolute_path = os.path.join(mocks_dir, filename)
        if os.path.isfile(absolute_path):
            with open(absolute_path, 'r') as f:
                mocks[filename] = json.load(f)

    return mocks
