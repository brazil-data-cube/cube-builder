#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Create a python click context and inject it to the global flask commands."""

import click
from bdc_db.cli import create_cli
from bdc_db.cli import create_db as bdc_create_db
from bdc_db.models import db
from flask.cli import with_appcontext
from flask_migrate.cli import db as flask_migrate_db

from . import create_app
from .config import Config

# Create cube-builder cli from bdc-db
cli = create_cli(create_app=create_app)


@flask_migrate_db.command()
@with_appcontext
@click.pass_context
def create_db(ctx: click.Context):
    """Create database. Make sure the variable SQLALCHEMY_DATABASE_URI is set."""
    ctx.forward(bdc_create_db)

    click.secho('Creating schema {}...'.format(Config.ACTIVITIES_SCHEMA), fg='green')
    with db.session.begin_nested():
        db.session.execute('CREATE SCHEMA IF NOT EXISTS {}'.format(Config.ACTIVITIES_SCHEMA))

    db.session.commit()


@cli.command('load-data')
@with_appcontext
def load_data():
    """Create Cube Builder composite functions supported."""
    from bdc_db.models import CompositeFunctionSchema, TemporalCompositionSchema, db
    from .utils import get_or_create_model

    with db.session.begin_nested():
        _, _ = get_or_create_model(
            CompositeFunctionSchema,
            defaults=dict(id='MED', description='Median by pixels'),
            id='MED'
        )

        _, _ = get_or_create_model(
            CompositeFunctionSchema,
            defaults=dict(id='STK', description='Best pixel'),
            id='STK'
        )

        _, _ = get_or_create_model(
            CompositeFunctionSchema,
            defaults=dict(id='IDENTITY', description=''),
            id='IDENTITY'
        )

        _, _ = get_or_create_model(
            TemporalCompositionSchema,
            defaults=dict(id='Anull', temporal_composite_unit='', temporal_schema='', temporal_composite_t=''),
            id='Anull'
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
@click.option('--composite-functions', type=click.STRING, help='Generate other composite functions (Must be registered)')
@click.option('--force', '-f', is_flag=True, help='Build data cube without cache')
@click.option('--with-rgb', is_flag=True, help='Generate a file with RGB bands, based in quick look.')
@with_appcontext
def build(datacube: str, collections: str, tiles: str, start: str, end: str, composite_functions=None,
          force=False, with_rgb=False):
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
        with_rgb=with_rgb
    )

    if composite_functions:
        data['composite_functions'] = composite_functions.split(',')

    parser = DataCubeProcessParser()
    parsed_data = parser.load(data)

    click.secho('Triggering data cube generation...', fg='green')
    res = CubeBusiness.maestro(**parsed_data)

    assert res['ok']


def main(as_module=False):
    """Load cube-builder as package in python module."""
    import sys
    cli.main(args=sys.argv[1:], prog_name="python -m cube_builder" if as_module else None)
