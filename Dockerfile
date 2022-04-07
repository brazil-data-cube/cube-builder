#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

FROM python:3.8

ARG GIT_COMMIT

LABEL "org.brazildatacube.maintainer"="Brazil Data Cube <brazildatacube@inpe.br>"
LABEL "org.brazildatacube.title"="Docker image for Data Cube Builder Service"
LABEL "org.brazildatacube.git_commit"="${GIT_COMMIT}"

ADD . /app

WORKDIR /app

RUN python3 -m pip install pip --upgrade setuptools wheel && \
    python3 -m pip install -e .[rabbitmq]
