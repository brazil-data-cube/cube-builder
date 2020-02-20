"""
Brazil Data Cube Scripts

Creates a python click context and inject it to the global flask commands

It allows to call own
"""

from multiprocessing import cpu_count
from bdc_db.models import db
from flask.cli import FlaskGroup, with_appcontext
from flask_migrate.cli import db as flask_migrate_db
from sqlalchemy_utils.functions import create_database, database_exists
import click

from cube_builder import create_app
from .config import Config


def create_cli(create_app=None):
    """
    Wrapper creation of Flask App in order to attach into flask click

    Args:
         create_app (function) - Create app factory (Flask)
    """
    def create_cli_app(info):
        if create_app is None:
            info.create_app = None

            app = info.load_app()
        else:
            app = create_app()

        return app

    @click.group(cls=FlaskGroup, create_app=create_cli_app)
    def cli(**params):
        """Command line interface for bdc_collection_builder"""
        pass

    return cli


cli = create_cli(create_app=create_app)


@flask_migrate_db.command()
@with_appcontext
def create():
    """Create database. Make sure the variable SQLALCHEMY_DATABASE_URI is set."""
    click.secho('Creating database {0}'.format(db.engine.url),
                fg='green')
    if not database_exists(str(db.engine.url)):
        create_database(str(db.engine.url))

    click.secho('Creating extension postgis...', fg='green')
    with db.session.begin_nested():
        db.session.execute('CREATE EXTENSION IF NOT EXISTS postgis')
        db.session.execute('CREATE SCHEMA IF NOT EXISTS {}'.format(Config.ACTIVITIES_SCHEMA))

    db.session.commit()


@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@with_appcontext
@click.pass_context
def worker(ctx):
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
    """Loads cube-builder as package in python module."""
    import sys
    cli.main(args=sys.argv[1:], prog_name="python -m cube_builder" if as_module else None)
