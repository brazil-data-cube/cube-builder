#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define a structure component to run celery worker."""

# Python Native
import logging

# 3rdparty
from celery.signals import worker_shutdown

# Builder
from cube_builder import create_app
from cube_builder.celery import create_celery_app


app = create_app()
celery = create_celery_app(app)
