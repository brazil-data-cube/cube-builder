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

"""Test the API cube_builder.views."""

import datetime

from click.testing import CliRunner
from dateutil.relativedelta import relativedelta
from flask import Response

from cube_builder import __version__
from cube_builder.cli import cli


def _assert_json_request(response: Response, status_code=200):
    assert response.status_code == status_code
    assert response.is_json

    return response.json


def test_index_api(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json['description'] == 'Cube Builder' and response.json['version'] == __version__


def test_create_grid(client, json_data):
    response = client.post('/create-grids', json=json_data['grid-bdc-md.json'])
    assert response.status_code == 201


def test_get_grid(client, json_data):
    response = client.get('/grids')
    grids = _assert_json_request(response)

    assert len(grids) > 0


def test_load_initial_data():
    res = CliRunner().invoke(cli, args=['load-data'])
    assert res.exit_code == 0


def test_create_cube(client, json_data):
    json_cube = json_data['lc8-16d-stk.json']
    response = client.post('/cubes', json=json_cube)

    cubes = _assert_json_request(response, status_code=201)
    assert isinstance(cubes, list)
    # TODO: validate json response with jsonschema api


def test_list_composite_functions(client):
    response = client.get('/composite-functions')

    data = _assert_json_request(response, status_code=200)
    for fn in data:
        assert fn['alias'] in ['STK', 'MED', 'IDT', 'LCF']


def test_list_periods_continuous_month(client):
    data = dict(
        start_date='2020-01-01',
        last_date='2020-12-31',
        step=1,
        schema='continuous',
        unit='month'
    )

    response = client.post('/list-periods', json=data)
    periods = _assert_json_request(response, status_code=200)
    assert len(periods['timeline']) == 12
    ref_date = datetime.date.fromisoformat(data['start_date'])
    for period_start, period_end in periods['timeline']:
        start = datetime.date.fromisoformat(period_start)
        end = datetime.date.fromisoformat(period_end)
        assert start == ref_date
        offset = relativedelta(day=31)
        assert end == (ref_date + offset)

        ref_date += offset + relativedelta(days=1)


def test_datacube_status(client, json_data):
    json_cube = json_data['lc8-16d-stk.json']
    identifier = f"{json_cube['datacube']}-{json_cube['version']}"
    response = client.get('/cube-status', query_string={'cube_name': identifier})
    _assert_json_request(response, status_code=200)

    # Test invalid request
    response = client.get('/cube-status')
    _assert_json_request(response, status_code=400)


def test_list_cubes(client):
    cube_info = _get_first_cube(client)

    response = client.get(f'/cubes/{cube_info["id"]}')
    cube = _assert_json_request(response, 200)
    assert cube['name'] == cube_info['name']


def test_update_cube_meta(client):
    cube = _get_first_cube(client)

    props = dict(
        title="New Cube - Updated",
        public=True
    )
    response = client.put(f'/cubes/{cube["id"]}', json=props)
    res = _assert_json_request(response, 200)
    assert res['message'] == 'Updated cube!'
    updated_cube = _get_first_cube(client)

    assert updated_cube['title'] == props['title']
    assert updated_cube['is_public'] == props['public']

    # invalid parameter
    props['public'] = 'invalid'
    response = client.put(f'/cubes/{cube["id"]}', json=props)
    _assert_json_request(response, 400)


def test_list_cube_tiles(client):
    cube = _get_first_cube(client)

    response = client.get(f'/cubes/{cube["id"]}/tiles')
    _assert_json_request(response, 200)


def _get_first_cube(client):
    response = client.get('/cubes')
    cubes = _assert_json_request(response, 200)
    assert len(cubes) > 0
    return cubes[0]
