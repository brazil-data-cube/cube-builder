#!/bin/bash

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

echo
echo "BUILD STARTED"
echo

if [ -z "${TAG_CUBE_BUILDER}" ]; then
  echo "NEW TAG CUBE-BUILDER:"
  read TAG_CUBE_BUILDER

  echo
fi

export IMAGE_CUBE_BUILDER="registry.dpi.inpe.br/brazil-data-cube/cube-builder"
export IMAGE_CUBE_BUILDER_FULL="${IMAGE_CUBE_BUILDER}:${TAG_CUBE_BUILDER}"
echo "IMAGE Cube Builder :: ${IMAGE_CUBE_BUILDER_FULL}"

docker-compose build

docker image tag ${IMAGE_CUBE_BUILDER}:latest ${IMAGE_CUBE_BUILDER_FULL}
docker push ${IMAGE_CUBE_BUILDER_FULL}
