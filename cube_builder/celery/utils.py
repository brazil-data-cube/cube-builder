#
# This file is part of Brazil Data Cube Collection Builder.
# Copyright (C) 2019-2020 INPE.
#
# Brazil Data Cube Collection Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Defines the utility functions to use among celery tasks."""
import requests

from celery import current_app
from urllib.parse import urlparse
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
