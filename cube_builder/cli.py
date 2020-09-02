#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Create a python click context and inject it to the global flask commands."""

import click
from bdc_catalog.models import CompositeFunction, db
from flask.cli import with_appcontext, FlaskGroup

from . import create_app


# Create cube-builder cli from bdc-db
@click.group(cls=FlaskGroup, create_app=create_app)
def cli():
    """Command line for data cube builder."""


@cli.command('load-data')
@with_appcontext
def load_data():
    """Create Cube Builder composite functions supported."""
    from .utils import get_or_create_model

    with db.session.begin_nested():
        _, _ = get_or_create_model(
            CompositeFunction,
            defaults=dict(name='Median', alias='MED', description='Median by pixels'),
            alias='MED'
        )

        _, _ = get_or_create_model(
            CompositeFunction,
            defaults=dict(name='Stack', alias='STK', description='Best pixel'),
            id='STK'
        )

        _, _ = get_or_create_model(
            CompositeFunction,
            defaults=dict(name='Identity', description=''),
            alias='IDT'
        )

    db.session.commit()


@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@with_appcontext
@click.pass_context
def worker(ctx: click.Context):
    """Run cube builder worker and make it available to execute data cube tasks.

    Uses celery default variables
    """
    from celery.bin.celery import main as _main
    from .celery import worker

    # TODO: Retrieve dynamically
    worker_context = '{}:celery'.format(worker.__name__)

    args = ["celery", "worker", "-A", worker_context]
    args.extend(ctx.args)

    _main(args)


@cli.command()
@click.argument('datacube')
@click.option('--collections', type=click.STRING, required=True, help='Collections to use')
@click.option('--tiles', type=click.STRING, required=True, help='Comma delimited tiles')
@click.option('--start', type=click.STRING, required=True, help='Start date')
@click.option('--end', type=click.STRING, required=True, help='End date')
@click.option('--bands', type=click.STRING, help='Comma delimited bands to generate')
@click.option('--force', '-f', is_flag=True, help='Build data cube without cache')
@click.option('--with-rgb', is_flag=True, help='Generate a file with RGB bands, based in quick look.')
@click.option('--token', type=click.STRING, help='Token to access data from STAC.')
@with_appcontext
def build(datacube: str, collections: str, tiles: str, start: str, end: str, bands: str = None,
          force=False, with_rgb=False, **kwargs):
    """Build data cube through command line.

    Args:
        datacube - Data cube name to generate
        collections - Comma separated collections to use
        tiles - Comma separated tiles to use
        start - Data cube start date
        end - Data cube end date
        bands - Comma separated bands to generate
        force - Flag to build data cube without cache. Default is False
    """
    from .business import CubeBusiness
    from .parsers import DataCubeProcessParser

    data = dict(
        datacube=datacube,
        collections=collections.split(','),
        start_date=start,
        end_date=end,
        tiles=tiles.split(','),
        force=force,
        with_rgb=with_rgb,
        **kwargs
    )

    if bands:
        data['bands'] = bands.split(',')

    parser = DataCubeProcessParser()
    parsed_data = parser.load(data)

    click.secho('Triggering data cube generation...', fg='green')
    res = CubeBusiness.maestro(**parsed_data)

    assert res['ok']


def main(as_module=False):
    """Load cube-builder as package in python module."""
    import sys
    cli.main(args=sys.argv[1:], prog_name="python -m cube_builder" if as_module else None)
