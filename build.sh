#!/bin/bash

#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

echo
echo "BUILD STARTED"
echo

if [ -z "${TAG_CUBE_BUILDER}" ]; then
  echo "NEW TAG CUBE-BUILDER:"
  read TAG_CUBE_BUILDER

  echo
fi

export IMAGE_CUBE_BUILDER="registry.dpi.inpe.br/brazildatacube/cube-builder"
export IMAGE_CUBE_BUILDER_FULL="${IMAGE_CUBE_BUILDER}:${TAG_CUBE_BUILDER}"
echo "IMAGE Cube Builder :: ${IMAGE_CUBE_BUILDER_FULL}"

docker-compose build

docker tag ${IMAGE_CUBE_BUILDER}:latest ${IMAGE_CUBE_BUILDER_FULL}
docker push ${IMAGE_CUBE_BUILDER_FULL}
