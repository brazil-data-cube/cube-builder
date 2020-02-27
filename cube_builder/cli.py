#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Create a python click context and inject it to the global flask commands."""

from bdc_db.models import db
from bdc_db.cli import create_db as bdc_create_db
from flask.cli import with_appcontext
from flask_migrate.cli import db as flask_migrate_db
import click

from . import create_app
from .config import Config


# Create cube-builder cli from bdc-db
cli = create_cli(create_app=create_app)


@flask_migrate_db.command()
@with_appcontext
def create(ctx: click.Context):
    """Create database. Make sure the variable SQLALCHEMY_DATABASE_URI is set."""

    ctx.forward(bdc_create_db)

    click.secho('Creating schema {}...'.format(Config.ACTIVITIES_SCHEMA), fg='green')
    with db.session.begin_nested():
        db.session.execute('CREATE SCHEMA IF NOT EXISTS {}'.format(Config.ACTIVITIES_SCHEMA))

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
@with_appcontext
def build(datacube: str, collections: str, tiles: str, start: str, end: str, bands: str = None, force=False):
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
        force=force
    )

    if bands:
        data['bands'] = bands.split(',')

        if 'quality' not in data['bands']:
            raise RuntimeError('Quality band is required')

    parser = DataCubeProcessParser()
    parsed_data = parser.load(data)

    click.secho('Triggering data cube generation...', fg='green')
    res = CubeBusiness.maestro(**parsed_data)

    assert res['ok']


def main(as_module=False):
    """Load cube-builder as package in python module."""
    import sys
    cli.main(args=sys.argv[1:], prog_name="python -m cube_builder" if as_module else None)
