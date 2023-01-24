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
ARG GIT_COMMIT
ARG BASE_IMAGE=python:3.8
FROM ${BASE_IMAGE}

ARG GIT_COMMIT

LABEL "org.brazildatacube.maintainer"="Brazil Data Cube <brazildatacube@inpe.br>"
LABEL "org.brazildatacube.title"="Docker image for Data Cube Builder Service"
LABEL "org.brazildatacube.description"="Docker image for Data Cube Builder application."
LABEL "org.brazildatacube.git_commit"="${GIT_COMMIT}"

# Build arguments
ARG CUBE_BUILDER_VERSION="1.0.0a1"
ARG CUBE_BUILDER_INSTALL_PATH="/opt/cube-builder/${CUBE_BUILDER_VERSION}"

ADD . ${CUBE_BUILDER_INSTALL_PATH}

WORKDIR ${CUBE_BUILDER_INSTALL_PATH}

RUN python3 -m pip install pip --upgrade setuptools wheel && \
    python3 -m pip install -e .[rabbitmq] && \
    python3 -m pip install gunicorn

EXPOSE 5000

CMD ["gunicorn", "-w4", "--bind=0.0.0.0:5000", "cube_builder:create_app()"]
