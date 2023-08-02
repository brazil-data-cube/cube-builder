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

"""Define Cube Builder celery module initialization."""

import logging
import os

import flask
from bdc_catalog.models import db
# 3rdparty
from celery import Celery
from flask import Flask

from ..config import Config
from ..constants import to_bool

CELERY_TASKS = [
    'cube_builder.celery.tasks',
]

celery_app = None


def create_celery_app(flask_app: Flask) -> Celery:
    """Create a Celery object and tir the celery config to the Flask app config.

    Wrap all the celery tasks in the context of Flask application

    Args:
        flask_app (flask.Flask): Flask app

    Returns:
        Celery The celery app
    """
    celery = Celery(
        flask_app.import_name,
        broker=Config.RABBIT_MQ_URL
    )

    # Load tasks
    celery.autodiscover_tasks(CELERY_TASKS)

    # Set same config of Flask into Celery flask_app
    flask_conf = {
        key: value for key, value in flask_app.config.items()
        # Disable the following patterns for env vars due deprecation in celery 5.x+
        if not key.startswith('CELERY_') and not key.startswith('CELERYD_')
    }

    celery.conf.update(flask_conf)

    always_eager = flask_app.config.get('TESTING', False)
    celery.conf.update(dict(
        broker_connection_retry_on_startup=True,
        task_acks_late=to_bool(os.getenv('CELERY_ACKS_LATE', '1')),
        task_always_eager=always_eager,
        worker_prefetch_multiplier=Config.CELERYD_PREFETCH_MULTIPLIER,
        result_backend='db+{}'.format(flask_app.config.get('SQLALCHEMY_DATABASE_URI')),
    ))

    TaskBase = celery.Task

    class ContextTask(TaskBase):
        """Define celery base tasks which supports Flask SQLAlchemy."""

        abstract = True

        def __call__(self, *args, **kwargs):
            """Prepare SQLAlchemy inside flask app."""
            if not celery.conf.CELERY_ALWAYS_EAGER:
                with flask_app.app_context():
                    # Following example of Flask
                    # Just make sure the task execution is running inside flask context
                    # https://flask.palletsprojects.com/en/1.1.x/patterns/celery/

                    return TaskBase.__call__(self, *args, **kwargs)
            else:
                logging.warning('Not Call context Task')

        def after_return(self, status, retval, task_id, args, kwargs, einfo):
            """Define teardown task execution.

            Whenever task finishes, it must tear down our db session, since the Flask SQLAlchemy
            creates scoped session at startup.
            FMI: https://gist.github.com/twolfson/a1b329e9353f9b575131
            """
            with flask_app.app_context():
                if not isinstance(retval, Exception):
                    db.session.commit()
                else:
                    try:
                        db.session.rollback()
                    except BaseException:
                        logging.warning('Error rollback transaction')
                        pass

                if not celery.conf.CELERY_ALWAYS_EAGER:
                    db.session.remove()

    celery.Task = ContextTask

    return celery
