..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2020 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Using Cube Builder
==================

This section explains how to use the Cube Builder application to generate data cubes from Sentinel 2, Landsat 8 and CBERS collections.


If you have not read yet how to install or deploy the system, please refer to `INSTALL.rst <./INSTALL.rst>`_ or `DEPLOY.rst <./DEPLOY.rst>`_ documentation.


Creating Grid Schema
--------------------

First of all, you need to create a Grid schema. In this example, we will create a Grid for ``BRAZIL``.
In order to create a GrsSchema ``BRAZIL``, use the following command:


.. code-block:: shell

    curl --location \
         --request POST '127.0.0.1:5000/api/cubes/create-grs-schema' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "name": "BRAZIL",
            "description": "albers equal area - 250k by tiles Brazil",
            "projection": "aea",
            "meridian": -46,
            "degreesx": 1.5,
            "degreesy": 1,
            "bbox": "-73.9872354804,5.24448639569,-34.7299934555,-33.7683777809"
         }'

The response will have status code ``201`` and the body:

.. code-block:: shell

    {
        "description": "albers equal area - 250k by tiles Brazil",
        "crs": "+proj=aea +lat_1=10 +lat_2=-40 +lat_0=0 +lon_0=-46 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
        "id": "BRAZIL"
    }


.. note::

    Remember that the bounding box ``bbox`` order is defined by: ``west,north,east,south``.


Creating Raster Schema
----------------------

.. note::

    If you already have the raster size schema *BRAZIL-10*, *BRAZIL-30* and *BRAZIL-64* in your database, you can skip this step.


The following sections describe how to create Raster Size Schemas for different resolutions.


Resolution 10 meters (Sentinel 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    curl --location \
         --request POST 'http://127.0.0.1:5000/api/cubes/create-raster-schema' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "grs_schema": "BRAZIL",
            "resolution": "10",
            "chunk_size_x": 256,
            "chunk_size_y": 256
         }'


It will create a raster size schema ``BRAZIL-10``. The response will have status code ``201`` and the body:

.. code-block:: json

    {
        "raster_size_y": 11727,
        "raster_size_x": 15744,
        "chunk_size_t": 1.0,
        "id": "BRAZIL-10",
        "raster_size_t": 1.0
    }


Resolution 30 meters (Landsat-8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    curl --location \
         --request POST 'http://127.0.0.1:5000/api/cubes/create-raster-schema' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "grs_schema": "BRAZIL",
            "resolution": "30",
            "chunk_size_x": 256,
            "chunk_size_y": 256
         }'


It will create a raster size schema ``BRAZIL-30``. The response will have status code ``201`` and the body:

.. code-block:: json

    {
        "raster_size_y": 3909,
        "raster_size_x": 5248,
        "chunk_size_t": 1.0,
        "id": "BRAZIL-30",
        "raster_size_t": 1.0
    }


Resolution 64 meters (CBERS4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    curl --location \
         --request POST 'http://127.0.0.1:5000/api/cubes/create-raster-schema' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "grs_schema": "BRAZIL",
            "resolution": "64",
            "chunk_size_x": 256,
            "chunk_size_y": 256
         }'


It will create a raster size schema ``BRAZIL-64``. The response will have status code ``201`` and the body:

.. code-block:: json

    {
        "raster_size_y": 1832,
        "raster_size_x": 2460,
        "chunk_size_t": 1.0,
        "id": "BRAZIL-64",
        "raster_size_t": 1.0
    }


.. warning::

    If you try to insert a already registered raster size schema, the response will have status code ``409`` representing
    duplicated.


Creating Temporal Composition Schema
------------------------------------

.. note::

    If you already have a composition schemas *monthly* (``M1month``) and *16 day in year* (``A16day``) in your database, you can skip this step.


Period Monthly
~~~~~~~~~~~~~~

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


Period 16 day in year
~~~~~~~~~~~~~~~~~~~~~

Use the following command to create a temporal composition schema ``A16day``:

.. code-block:: shell

    curl --location --request POST '127.0.0.1:5000/api/cubes/create-temporal-schema' \
         --header 'Content-Type: application/json' \
        --data-raw '{
            "temporal_composite_unit": "day",
            "temporal_schema": "A",
            "temporal_composite_t": "16"
        }'

It will create a temporal composition schema ``A16day``. The response will have status code ``201`` and the body:

.. code-block:: json

    {
        "id": "A16day",
        "temporal_schema": "A",
        "temporal_composite_t": "16"
    }


.. warning::

    If you try to insert a already registered temporal composite schema, the response will have status code ``409`` representing
    duplicated.


Creating data cube Landsat-8
----------------------------

In order to create data cube Landsat-8, use the following command to create data cube metadata:

.. code-block:: shell

    curl --location --request POST '127.0.0.1:5000/api/cubes/create' \
         --header 'Content-Type: application/json' \
         --data-raw '{
             "datacube": "LC8_30_1M",
             "grs": "BRAZIL",
             "resolution": 30,
             "temporal_schema": "M1month",
             "bands_quicklook": ["swir2", "nir", "red"],
             "composite_function_list": ["MEDIAN", "STACK"],
             "bands": ["coastal", "blue", "green", "red", "nir", "swir1", "swir2", "evi", "ndvi", "quality", "cnc"],
             "description": "Landsat 8 30m - Monthly"
         }'


Trigger data cube generation with following command:

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
            "end_date": "2019-01-31"
         }'


.. note::

    The command line ``cube-builder build`` has few optional parameters such
    ``bands``, which defines bands to generate data cube.


Creating data cube Sentinel 2
-----------------------------

In order to create data cube Sentinel 2, use the following command to create data cube metadata:

.. code-block:: shell

    # Using curl (Make sure to execute cube-builder run)
    curl --location --request POST '127.0.0.1:5000/api/cubes/create' \
            --header 'Content-Type: application/json' \
            --data-raw '{
                "datacube": "S2_10_1M",
                "grs": "BRAZIL",
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


Creating data cube CBERS4 AWFI
------------------------------

In order to create data cube CBERS4 AWFI, use the following command to create data cube metadata:

.. code-block:: shell

    # Using curl (Make sure to execute cube-builder run)
    curl --location --request POST '127.0.0.1:5000/api/cubes/create' \
            --header 'Content-Type: application/json' \
            --data-raw '{
                "datacube": "C4_64_1M",
                "grs": "BRAZIL",
                "resolution": 64,
                "temporal_schema": "M1month",
                "bands_quicklook": ["red", "nir", "green"],
                "composite_function_list": ["MEDIAN", "STACK"],
                "bands": ["blue", "green", "red", "nir", "evi", "ndvi", "quality", "cnc"],
                "description": "CBERS4 AWFI - Monthly"
            }'

Trigger data cube generation with following command:

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
