"""
Brazil Data Cube Scripts

Creates a python click context and inject it to the global flask commands

It allows to call own
"""

from multiprocessing import cpu_count
import click
from flask.cli import FlaskGroup, with_appcontext
from cube_builder import create_app


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


def main(as_module=False):
    # TODO omit sys.argv once https://github.com/pallets/click/issues/536 is fixed
    import sys
    cli.main(args=sys.argv[1:], prog_name="python -m cube_builder" if as_module else None)
