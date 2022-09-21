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

"""Create a python click context and inject it to the global flask commands."""

import click
from bdc_catalog.models import Application, CompositeFunction, db
from flask.cli import FlaskGroup, with_appcontext

from . import create_app
from .controller import CubeController
from .models import CubeParameters
from .version import __version__


@click.group(cls=FlaskGroup, create_app=create_app)
def cli():
    """Command line for data cube builder."""


@cli.command('load-data')
@with_appcontext
def load_data():
    """Create Cube Builder composite functions supported."""
    from .utils.processing import get_or_create_model

    with db.session.begin_nested():
        _, _ = get_or_create_model(
            CompositeFunction,
            defaults=dict(name='Median', alias='MED', description='Median by pixels'),
            alias='MED'
        )

        _, _ = get_or_create_model(
            CompositeFunction,
            defaults=dict(name='Least Cloud Cover First', alias='LCF', description='Best pixel'),
            alias='LCF'
        )

        _, _ = get_or_create_model(
            CompositeFunction,
            defaults=dict(name='Identity', description=''),
            alias='IDT'
        )

        where = dict(
            name=__package__,
            version=__version__
        )

        # Cube-Builder application
        application, _ = get_or_create_model(
            Application,
            defaults=dict(),
            **where
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
@click.option('--stac-url', type=click.STRING, help='STAC to search')
@click.option('--force', '-f', is_flag=True, help='Build data cube without cache')
@click.option('--token', type=click.STRING, help='Token to access data from STAC.')
@click.option('--export-files', type=click.Path(writable=True), help='Export Identity Merges in file')
@with_appcontext
def build(datacube: str, collections: str, tiles: str, start: str, end: str, bands: str = None,
          stac_url: str = None, force=False, with_rgb=False, shape=None, export_files=None, **kwargs):
    """Build data cube through command line.

    Args:
        datacube - Data cube name to generate
        collections - Comma separated collections to use
        tiles - Comma separated tiles to use
        start - Data cube start date
        end - Data cube end date
        bands - Comma separated bands to generate
        force - Flag to build data cube without cache. Default is False
        with_rgb - Flag to RGB file using quicklook reference. Default is False.
        shape - Use custom output raster shape. i.e 10980x10980
    """
    from .forms import DataCubeProcessForm

    mask = kwargs.get('mask')

    if mask:
        kwargs['mask'] = eval(mask)

    data = dict(
        datacube=datacube,
        collections=collections.split(','),
        start_date=start,
        end_date=end,
        tiles=tiles.split(','),
        force=force,
        with_rgb=with_rgb,
        stac_url=stac_url,
        **kwargs
    )

    if bands:
        data['bands'] = bands.split(',')

    if shape is not None:
        shape = shape.split('x')

        if len(shape) != 2:
            raise RuntimeError(f'Expected 2d shape, but got {shape}')

        data['shape'] = shape

    parser = DataCubeProcessForm()
    parsed_data = parser.load(data)
    parsed_data['export_files'] = export_files

    click.secho('Triggering data cube generation...', fg='green')
    res = CubeController.maestro(**parsed_data)

    assert res['ok']


@cli.command('configure')
@click.argument('datacube')
@click.option('--stac-url', type=click.STRING, help='STAC to search')
@click.option('--reuse-from', type=click.STRING, help='Reuse data cube from another data cube.')
@click.option('--with-rgb', is_flag=True, help='Generate a file with RGB bands, based in quick look.')
@click.option('--token', type=click.STRING, help='Token to access data from STAC.')
@click.option('--shape', type=click.STRING, help='Use custom output shape. i.e `--shape=10980x10980`')
@click.option('--histogram-matching', is_flag=True, help='Match the histogram in the temporal composition function.')
@click.option('--mask',  type=click.STRING, help='Custom mask values for data cube.')
@click.option('--quality-band', type=click.STRING, help='Quality band name')
@click.option('--cloud-cover', type=click.FLOAT, help='Cloud Cover Factor. Default is 100 to use all.', default=100)
@with_appcontext
def _configure_parameters(datacube: str, **kwargs):
    """Configure the default parameters for data cube.

    DATACUBE must be a DatacubeName-Version.
    """
    cube = CubeController.get_cube_or_404(cube_full_name=datacube)

    if kwargs.get('mask'):
        kwargs['mask'] = eval(kwargs['mask'])

    cloud_cover = kwargs.pop('cloud_cover', 100)
    kwargs['stac_kwargs'] = dict(query={"eo:cloud_cover": {"lte": cloud_cover}})

    quality_band = kwargs['quality_band']
    band_map = [b.name for b in cube.bands]
    if quality_band not in band_map:
        raise RuntimeError(f'Invalid quality band "{quality_band}"')

    non_none_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    parameters = CubeController.configure_parameters(cube.id, **non_none_kwargs)

    click.secho(f'The parameters were set for data cube {datacube}.')
    _display_parameters(parameters['metadata_'])


@cli.command('show-parameters')
@click.argument('datacube')
@with_appcontext
def _show_parameters(datacube: str):
    """Display the data cube parameters.

    DATACUBE must be a DataCubeName-Version.
    """
    cube = CubeController.get_cube_or_404(cube_full_name=datacube)

    parameters = CubeParameters.query().filter(CubeParameters.collection_id == cube.id).first_or_404()

    _display_parameters(parameters.metadata_)


def _display_parameters(metadata: dict):
    for key, value in metadata.items():
        click.secho(f'\t{key} -> {value}')


def main(as_module=False):
    """Load cube-builder as package in python module."""
    import sys
    cli.main(args=sys.argv[1:], prog_name="python -m cube_builder" if as_module else None)
