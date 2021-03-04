#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

FROM registry.dpi.inpe.br/brazildatacube/geo:0.3

ADD . /app

WORKDIR /app

RUN pip3 install pip --upgrade && \
    pip install wheel && \
    pip install -e .
