#
# This file is part of Brazil Data Cube Collection Builder.
# Copyright (C) 2019-2021 INPE.
#
# Brazil Data Cube Collection Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Defines the utility functions to use among celery tasks."""

import logging
import os
import shutil
from urllib.parse import urlparse

import requests
from celery import current_app

from ..config import Config


def list_running_tasks():
    """List all running tasks in celery cluster."""
    inspector = current_app.control.inspect()

    return inspector.active()


def list_pending_tasks():
    """List all pending tasks in celery cluster."""
    inspector = current_app.control.inspect()

    return inspector.reserved()


def list_queues():
    """List all cube-builder queues from RabbitMQ."""
    url = urlparse(Config.RABBIT_MQ_URL)
    response = requests.get(f'http://{url.hostname}:{15672}/api/queues?columns=name,messages,'\
                            'messages_ready,messages_unacknowledged',
                            auth=(url.username, url.password))

    tasks = dict()

    for task in response.json():
        if 'cube' in task['name']:
            tasks[task['name']] = dict(total=task['messages'],
                                       ready=task['messages_ready'],
                                       unacked=task['messages_unacknowledged'])

    return tasks


def clear_merge(merge_date, scenes):
    """Clear entire directory containing the Data Cube Identity Item."""
    base_dir = None
    for band, absolute_path in scenes['ARDfiles'].items():
        if base_dir is None:
            base_dir = os.path.dirname(absolute_path)

        os.remove(absolute_path)

    if base_dir is not None:
        # Ensure folder is empty
        if not os.listdir(base_dir):
            shutil.rmtree(base_dir)
    logging.info(f'Cleaning up {merge_date} - {base_dir}')
