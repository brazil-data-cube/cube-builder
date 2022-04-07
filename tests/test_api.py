#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Test the API cube_builder.views."""

import datetime

from dateutil.relativedelta import relativedelta
from flask import Response


def _assert_json_request(response: Response, status_code=200):
    assert response.status_code == status_code
    assert response.is_json

    return response.json


def test_create_grid(client, json_data):
    response = client.post('/create-grids', json=json_data['grid-bdc-md.json'])
    assert response.status_code == 201


def test_get_grid(client, json_data):
    response = client.get('/grids')
    grids = _assert_json_request(response)

    assert len(grids) > 0


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
