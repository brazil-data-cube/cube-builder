..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2020 INPE.

    Cube Builder free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Deploying
=========

The ``docker-compose.yml`` in the root of the source tree can be used to run cube-builder as a multi-container application.


This section explains how to get the cube-builder service up and running with Docker and Docker Compose.
If you do not have Docker installed, take a look at `this tutorial on how to install it in your system <https://docs.docker.com/install/>`_.
See also the `tutorial on how to install Docker Compose <https://docs.docker.com/compose/install/>`_.


Building the Docker Image and Launching cube-builder
----------------------------------------------------

Use the following command in order to launch all the containers needed to run cube-builder [#f1]_:

.. code-block:: shell

    $ docker-compose up -d


If the above command runs successfully, you will be able to list the launched containers:

.. code-block:: shell

    $ docker container ls

    CONTAINER ID        IMAGE                                                    COMMAND                  CREATED             STATUS              PORTS                    NAMES
    a3bb86d2df56        rabbitmq:3-management                                    "docker-entrypoint.s…"   3 minutes ago       Up 2 minutes        4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp   cube-builder-rabbitmq
    e3862ab6e756        registry.dpi.inpe.br/brazildatacube/cube-builder:latest  "bash -c 'cube-build…"   2 minutes ago       Up 2 minutes        0.0.0.0:5001->5000/tcp   cube-builder-api
    13caa0f27030        registry.dpi.inpe.br/brazildatacube/cube-builder:latest  "cube-builder worker…"   2 minutes ago       Up 2 minutes                                 cube-builder-worker



Temporal Composition Schema
---------------------------

A Temporal Composition Schema is used to describe how the data cube will be created.

You can define a data cube with temporal schema ``monthly``, ``annual`` with interval of 16 days, ``seasonal``, etc. Once defined,
the ``cube_builder`` will seek for all images within period given and will generate data cube passing these images to a composite function.


Creating temporal composition schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    If you already have a composition schema *monthly* (``M1month``) in your database, you can skip this step.

Use the following command to create a temporal composition schema ``Monthly``:

.. code-block:: shell

        curl --location --request POST '127.0.0.1:5000/api/cubes/create-temporal-schema' \
             --header 'Content-Type: application/json' \
            --data-raw '{
                "temporal_composite_unit": "month",
                "temporal_schema": "M",
                "temporal_composite_t": "1"
            }'

It will create a temporal composition schema ``M1month``. The response will have status code ``201`` and the body:

.. code-block:: json

    {
        "id": "M1month",
        "temporal_schema": "M",
        "temporal_composite_t": "1"
    }


.. warning::::

    If you try to insert a already registered temporal composite schema, the response will have status code ``409`` representing
    duplicated.


Creating datacube Landsat8
--------------------------

Create datacube metadata

.. code-block:: shell

        curl --location --request POST '127.0.0.1:5000/api/cubes/create' \
             --header 'Content-Type: application/json' \
             --data-raw '{
                 "datacube": "LC8_30_1M",
                 "grs": "aea_250k",
                 "resolution": 30,
                 "temporal_schema": "M1month",
                 "bands_quicklook": ["swir2", "nir", "red"],
                 "composite_function_list": ["MEDIAN", "STACK"],
                 "bands": ["coastal", "blue", "green", "red", "nir", "swir1", "swir2", "evi", "ndvi", "quality", "cnc"],
                 "description": "Landsat 8 30m - Monthly"
             }'


Trigger datacube generation with following command:

.. code-block:: shell

        # Using cube-builder command line
        cube-builder build LC8_30_1M_MED \
            --collections=LC8SR \
            --tiles=089098 \
            --start=2019-01-01 \
            --end=2019-01-31

        # Using curl (Make sure to execute cube-builder run)
        curl --location \
             --request POST '127.0.0.1:5000/api/cubes/process' \
             --header 'Content-Type: application/json' \
             --data-raw '{
                "datacube": "LC8_30_1M_MED",
                "collections": ["LC8SR"],
                "tiles": ["089098"],
                "start_date": "2019-01-01",
                "end_date": "2019-01-31",
                "bands": ["swir2", "nir", "red", "evi", "quality"]
             }'


.. note::

    The command line ``cube-builder build`` has few optional parameters such
    ``bands``, which defines bands to generate data cube.


Creating datacube Sentinel-2
----------------------------

Use the following code to create data cube metadata of Sentinel 2:

.. code-block:: shell

    # Using curl (Make sure to execute cube-builder run)
    curl --location --request POST '127.0.0.1:5000/api/cubes/create' \
            --header 'Content-Type: application/json' \
            --data-raw '{
                "datacube": "S2_10_1M",
                "grs": "aea_250k",
                "resolution": 10,
                "temporal_schema": "M1month",
                "bands_quicklook": ["swir2", "nir", "red"],
                "composite_function_list": ["MEDIAN", "STACK"],
                "bands": [
                    "coastal",
                    "blue",
                    "green",
                    "red",
                    "redge1",
                    "redge2",
                    "redge3",
                    "nir",
                    "bnir",
                    "swir1",
                    "swir2",
                    "ndvi",
                    "evi",
                    "quality",
                    "cnc"
                ],
                "description": "Sentinel 2 10m - Monthly"
            }'


Trigger datacube generation with following command:

.. code-block:: shell

    # Using cube-builder command line
    cube-builder build S2_10_1M_MED \
        --collections=S2SR_SEN28 \
        --tiles=089098 \
        --start=2019-01-01 \
        --end=2019-01-31


Creating datacube CBERS4 AWFI
-----------------------------

Use the following code to create data cube metadata of CBERS 4 AWFI:

.. code-block:: shell

    # Using curl (Make sure to execute cube-builder run)
    curl --location --request POST '127.0.0.1:5000/api/cubes/create' \
            --header 'Content-Type: application/json' \
            --data-raw '{
                "datacube": "C4_64_1M",
                "grs": "aea_250k",
                "resolution": 64,
                "temporal_schema": "M1month",
                "bands_quicklook": ["red", "nir", "green"],
                "composite_function_list": ["MEDIAN", "STACK"],
                "bands": ["blue", "green", "red", "nir", "evi", "ndvi", "quality", "cnc"],
                "description": "CBERS4 AWFI - Monthly"
            }'

Trigger datacube generation with following command:

.. code-block:: shell

    # Using cube-builder command line
    cube-builder build C4_64_1M_MED \
        --collections=CBERS4_AWFI_L4_SR \
        --tiles=089098 \
        --start=2019-01-01 \
        --end=2019-01-31


.. note::

    In order to restart data cube generation, just pass the same command line to trigger a data cube.
    It will reuse the entire process, executing only the failed tasks. You can also pass optional parameter
    ``--force`` to build data cube without cache.


.. rubric:: Footnotes

.. [#f1]

    | For now you will need to login into the BDC registry:
    | ``$ docker login registry.dpi.inpe.br``
    |
    | In the next releases we will get ride of this internal registry.