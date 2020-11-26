..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2020 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Using Cube Builder
==================

This section explains how to use the Cube Builder application to generate data cubes from Sentinel 2, Landsat 8 and CBERS collections.


If you have not read yet how to install or deploy the system, please refer to `INSTALL.rst <./INSTALL.rst>`_ or `DEPLOY.rst <DEPLOY.rst>`_ documentation.


Creating a Grid for the Data Cubes
----------------------------------


A Data Cube must have an associated grid. The example showed below will create a grid over the Brazil bounds, named ``BRAZIL``:


.. code-block:: shell

    curl --location \
         --request POST '127.0.0.1:5000/create-grs' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "name": "BRAZIL",
            "description": "albers equal area - 250k by tiles Brazil",
            "projection": "aea",
            "meridian": -54,
            "degreesx": 1.5,
            "degreesy": 1,
            "bbox": "-73.9872354804,5.24448639569,-34.7299934555,-33.7683777809"
         }'


The response will have status code ``201`` and the body:


.. code-block:: shell

    {
        "Grid BRAZIL created with successfully"
    }


.. note::

    Remember that the bounding box ``bbox`` order is defined by: ``west,north,east,south``.



Creating a Temporal Composition Schema
--------------------------------------


.. note::

    If you already have a composition schemas *monthly* (``M1month``) and *16 day in year* (``A16day``) in your database, you can skip this step.


Period Monthly
**************

Use the following command to create a temporal composition schema ``Monthly``:

.. code-block:: shell

    curl --location --request POST '127.0.0.1:5000/create-temporal-schema' \
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


Period 16 day, recycled by year
*******************************


Use the following command to create a temporal composition schema ``A16day``:


.. code-block:: shell

    curl --location --request POST '127.0.0.1:5000/create-temporal-schema' \
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

    If you try to insert a already registered temporal composite schema, the response will have status code ``409`` representing duplicated.


Creating the Definition of Landsat-8 based Data Cube
----------------------------------------------------


In order to create data cube Landsat-8, use the following command to create data cube metadata:


.. code-block:: shell

    curl --location --request POST '127.0.0.1:5000/create-cube' \
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
         --request POST '127.0.0.1:5000/start-cube' \
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
    curl --location --request POST '127.0.0.1:5000/create-cube' \
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
    curl --location --request POST '127.0.0.1:5000/create-cube' \
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
        --collections=CBERS4_AWFI_L4_SR-1 \
        --tiles=089098 \
        --start=2019-01-01 \
        --end=2019-01-31


.. note::

    In order to restart data cube generation, just pass the same command line to trigger a data cube.
    It will reuse the entire process, executing only the failed tasks. You can also pass optional parameter
    ``--force`` to build data cube without cache.
