..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2020 INPE.

    Cube Builder free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Prepare and initialize database
-------------------------------

.. note::

    The ``cube-builder`` uses `bdc-db <https://github.com/brazil-data-cube/bdc-db/>`_ as database definition to store data cube metadata.
    Make sure you have a prepared database on PostgreSQL. You can follow steps `here <https://github.com/brazil-data-cube/bdc-db/blob/master/RUNNING.rst>`_.


Edit file **cube_builder/config.py** the following variables:

1. **SQLALCHEMY_DATABASE_URI** URI Connection to database
2. **DATA_DIR** Path to store datacubes

.. code-block:: shell

        cube-builder db create # Create database and schema
        cube-builder db upgrade # Up migrations


Running http server and worker
------------------------------

Once everything configured, run local server:

.. code-block:: shell

        cube-builder run


After that, run local celery worker:

.. code-block:: shell

        cube-builder worker -l INFO --concurrency 8


You may need to replace the definition of some parameters:

    - ``-l INFO``: defines the ``Logging level``. You may choose between ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``, or ``FATAL``.
    - ``--concurrency 8``: defines the number of concurrent processes to generate of data cube. The default is the number of CPUs available on your system.


.. note::

    The command line ``cube-builder worker`` is an auxiliary tool that wraps celery command line using ``cube_builder`` as context.
    In this way, all ``celery worker`` parameters currently supported. See more in `Celery Workers Guide <https://docs.celeryproject.org/en/stable/userguide/workers.html>`_.


.. warning::

    **Beware**: The ``cube-builder`` may use much memory for each concurrent process, since it opens multiple image collection in memory.


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
                 "description": "Landsat8 Cubes 30m - Monthly"
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
                "description": "S2 10m Monthly"
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
