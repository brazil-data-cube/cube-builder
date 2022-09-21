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
