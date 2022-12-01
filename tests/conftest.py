#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
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
