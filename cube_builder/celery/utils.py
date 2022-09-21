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

"""Defines the utility functions to use among celery tasks."""

import logging
import os
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
        basename = os.path.basename(base_dir)

        if merge_date not in basename:
            logging.warning(f'Skipping clear merge {base_dir} - {merge_date}')
            continue

        os.remove(absolute_path)

    if base_dir is not None:
        # Ensure folder is empty
        if not os.listdir(base_dir):
            os.rmdir(base_dir)
    logging.info(f'Cleaning up {merge_date} - {base_dir}')
