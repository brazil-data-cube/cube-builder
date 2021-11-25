#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define a structure component to run celery worker."""
import os

from celery.signals import celeryd_after_setup

# Builder
from cube_builder import create_app
from cube_builder.celery import create_celery_app

app = create_app(os.getenv('FLASK_ENV', 'production'))
celery = create_celery_app(app)


@celeryd_after_setup.connect()
def load_models(*args, **kwargs):
    """Load celery models when worker is ready."""
    from celery.backends.database import SessionManager

    session = SessionManager()
    engine = session.get_engine(celery.backend.url)
    session.prepare_models(engine)
