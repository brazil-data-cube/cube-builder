#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

FROM python:3.8

ADD . /app

WORKDIR /app

RUN python -m pip install pip --upgrade && \
    python -m pip install wheel && \
    python -m pip install -e .[rabbitmq]
