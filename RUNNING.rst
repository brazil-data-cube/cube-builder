..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019 INPE.

    Cube Builder free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Prepare and initialize database
-------------------------------

Edit file **cube_builder/config.py** the following variables:

1. **SQLALCHEMY_DATABASE_URI** URI Connection to database
2. **DATA_DIR** Path prefix to remove from asset and store on database
3. **ACTIVITIES_SCHEMA** Schema of datacube activities. Default is **cube_builder**

.. code-block:: shell

        cube-builder db create # Create database and schema
        cube-builder db upgrade # Up migrations


Running http server
-------------------

Once everything configured, run local server:

.. code-block:: shell

        cube-builder run


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
                 "bands": {
                     "names": ["coastal", "blue", "green", "red", "nir", "swir1", "swir2", "evi", "ndvi", "quality"],
                     "min": 0,
                     "max": 10000,
                     "fill": -9999,
                     "scale": 0.0001,
                     "data_type": "Int16"
                 },
                 "description": "Landsat8 Cubes 30m - Monthly"
             }'


Trigger datacube generation with following command:

.. code-block:: shell

        # Using cube-builder command line
        cube-builder build LC8_30_1M_MED --collections=LC8SR --tiles=089098 --start=2019-01-01 --end=2019-01-31 --bands=swir2,nir,red,evi,quality
        # Using curl (Make sure to execute cube-builder run)
        curl --location \
             --request POST '127.0.0.1:5000/api/cubes/process' \
             --header 'Content-Type: application/json'
             --data-raw '{
                "datacube": "LC8_30_1M_MED",
                "collections": ["LC8SR"],
                "tiles": ["089098"],
                "start_date": "2019-01-01",
                "end_date": "2019-01-31",
                "bands": ["swir2", "nir", "red", "evi", "quality"]
             }'