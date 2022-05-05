..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2021 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Using Cube Builder
==================

This section explains how to use the Cube Builder application to generate data cubes from Sentinel 2, Landsat 8 and CBERS collections.


If you have not read yet how to install or deploy the system, please refer to `INSTALL.rst <./INSTALL.rst>`_ or `DEPLOY.rst <DEPLOY.rst>`_ documentation.
You may also read the `Brazil Data Cube Documentation <https://brazil-data-cube.github.io/>`_ for further details about Brazil Data Cube Project.


Creating a Grid for the Data Cubes
----------------------------------


A Data Cube must have an associated grid as mentioned in `BDC Grid <https://brazil-data-cube.github.io/products/specifications/bdc-grid.html?highlight=grid>`_.
For this example, we will create 3 grids for different data cubes `BRAZIL_LG`, `BRAZIL_MD` and `BRAZIL_SM`.

The grid ``BRAZIL_LG`` will be used by collections `CBERS4` which has resolution `64 meters`::

    curl --location \
         --request POST '127.0.0.1:5000/create-grids' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "name": "BRAZIL_LG",
            "description": "Brazil Large Grid - Albers Equal Area",
            "projection": "aea",
            "meridian": -54,
            "degreesx": 6,
            "degreesy": 4,
            "bbox": [-73.9872354804, -33.7683777809, -34.7299934555, 5.24448639569],
            "srid": 100001
         }'

The response will have status code ``201`` and the body::

    "Grid BRAZIL_LG created."


The grid ``BRAZIL_MD`` will be used by collection `Landsat-8` which has resolution `30 meters`::

    curl --location \
         --request POST '127.0.0.1:5000/create-grids' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "name": "BRAZIL_MD",
            "description": "Brazil Medium Grid - Albers Equal Area",
            "projection": "aea",
            "meridian": -54,
            "degreesx": 3,
            "degreesy": 2,
            "bbox": [-73.9872354804, -33.7683777809, -34.7299934555, 5.24448639569],
            "srid": 100001
         }'

The response will have status code ``201`` and the body::

    "Grid BRAZIL_MD created."


The grid ``BRAZIL_SM`` will be used by collection `Sentinel-2` which has resolution `10 meters`::

    curl --location \
         --request POST '127.0.0.1:5000/create-grids' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "name": "BRAZIL_SM",
            "description": "Brazil Medium Grid - Albers Equal Area",
            "projection": "aea",
            "meridian": -54,
            "degreesx": 1.5,
            "degreesy": 1,
            "bbox": [-73.9872354804, -33.7683777809, -34.7299934555, 5.24448639569],
            "srid": 100001
         }'

The response will have status code ``201`` and the body::

    "Grid BRAZIL_SM created."


.. note::

    Remember that the bounding box ``bbox`` order is defined by: ``west,north,east,south``.

    You may also change ``degreesx`` and ``degreesy`` if you would like to increase or decrease the Grid Tile size.


Creating the Definition of Landsat-8 based Data Cube
----------------------------------------------------

In order to create data cube Landsat-8 monthly using the composite function ``Stack`` (`LC8_30_1M_STK`), use the following command to create data cube metadata::

    curl --location \
         --request POST '127.0.0.1:5000/cubes' \
         --header 'Content-Type: application/json' \
         --data-raw '
    {
        "datacube": "LC8",
        "grs": "BRAZIL_MD",
        "title": "Landsat-8 (OLI) Cube Stack Monthly - v001",
        "resolution": 30,
        "version": 1,
        "metadata": {
            "license": "MIT",
            "platform": {
                "code": "Landsat-8",
                "instruments": "OLI"
            }
        },
        "temporal_composition": {
            "schema": "Continuous",
            "step": 1,
            "unit": "month"
        },
        "composite_function": "LCF",
        "bands_quicklook": [
            "sr_band7",
            "sr_band5",
            "sr_band4"
        ],
        "bands": [
            {"name": "sr_band1", "common_name": "coastal", "data_type": "int16"},
            {"name": "sr_band2", "common_name": "blue", "data_type": "int16"},
            {"name": "sr_band3", "common_name": "green", "data_type": "int16"},
            {"name": "sr_band4", "common_name": "red", "data_type": "int16"},
            {"name": "sr_band5", "common_name": "nir", "data_type": "int16"},
            {"name": "sr_band6", "common_name": "swir1", "data_type": "int16"},
            {"name": "sr_band7", "common_name": "swir2", "data_type": "int16"},
            {"name": "Fmask4", "common_name": "quality", "data_type": "uint8"}
        ],
        "indexes": [
            {
                "name": "EVI",
                "common_name": "evi",
                "data_type": "int16",
                "metadata": {
                    "expression": {
                        "bands": ["sr_band5", "sr_band4", "sr_band2"],
                        "value": "(10000. * 2.5 * (sr_band5 - sr_band4) / (sr_band5 + 6. * sr_band4 - 7.5 * sr_band2 + 10000.))"
                    }
                }
            },
            {
                "name": "NDVI",
                "common_name": "ndvi",
                "data_type": "int16",
                "metadata": {
                    "expression": {
                        "bands": ["sr_band5", "sr_band4"],
                        "value": "10000. * ((sr_band5 - sr_band4)/(sr_band5 + sr_band4))"
                    }
                }
            }
        ],
        "quality_band": "Fmask4",
        "description": "This datacube contains the all available images from Landsat-8, with 30 meters of spatial resolution, reprojected and cropped to BDC_MD grid, composed each 16 days using the best pixel (Stack) composite function.",
        "parameters": {
            "mask": {
                "clear_data": [0, 1],
                "not_clear_data": [2, 3, 4],
                "nodata": 255,
                "saturated_data": []
            }
        }
    }'

.. note::

    If you would like to create a data cube with temporal composition with ``16 days`` which reset the time line per year, you may change the JSON key ``temporal_composition``::

        ..
        "temporal_composition": {
            "schema": "Cyclic",
            "step": 16,
            "unit": "day",
            "cycle": {
                "step": 1,
                "unit": "year"
            }
        }
        ..

    The property ``mask`` inside ``parameters`` represents how the Cube Builder will deal with ``Clear Data`` and ``Not Clear Data`` pixels.
    The ``Clear Data`` pixels are considered to identify the ``Best Pixel`` (Stack) and it is count on the ``Clear Observation Band`` (``ClearOb``).

In order to trigger a data cube, we are going to use a collection `LC8SR-1` made with Surface Reflectance using LaSRC 2.0 with cloud masking Fmask 4.2.

To trigger a data cube, use the following command::

    cube-builder build LC8_30_1M_LCF \
        --collections=LC8SR-1 \
        --tiles=044048 \
        --start=2019-01-01 \
        --end=2019-01-31

    # Using curl (Make sure to execute cube-builder run)
    curl --location \
         --request POST '127.0.0.1:5000/start-cube' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "datacube": "LC8_30_1M_LCF",
            "collections": ["LC8SR-1"],
            "tiles": ["044048"],
            "start_date": "2019-01-01",
            "end_date": "2019-01-31"
         }'


.. note::

    The command line ``cube-builder build`` has few optional parameters such
    ``bands``, which defines bands to generate data cube.

    You may also pass ``--stac-url=URL_TO_STAC`` (command line) or ``"stac_url": "URL_TO_STAC"`` (API only)
    if you would like to generate data cube using a different STAC provider. Remember that the ``--collection`` must exists.


Creating data cube Sentinel 2
-----------------------------

In order to create data cube Sentinel 2, use the following command to create data cube metadata:

.. code-block:: shell

    curl --location \
         --request POST '127.0.0.1:5000/cubes' \
         --header 'Content-Type: application/json' \
         --data-raw '
    {
        "datacube": "S2",
        "grs": "BRAZIL_SM",
        "title": "Sentinel-2 SR - LaSRC/Fmask 4.2 - Data Cube Stack 16 days -v001",
        "resolution": 10,
        "version": 1,
        "metadata": {
            "license": "MIT",
            "platform": {
                "code": "Sentinel-2",
                "instruments": "MSI"
            }
        },
        "temporal_composition": {
            "schema": "Cyclic",
            "step": 16,
            "unit": "day",
            "cycle": {
                "unit": "year",
                "step": 1
            }
        },
        "composite_function": "LCF",
        "bands_quicklook": [
            "sr_band12",
            "sr_band8a",
            "sr_band4"
        ],
        "bands": [
            {"name": "sr_band1", "common_name": "coastal", "data_type": "int16"},
            {"name": "sr_band2", "common_name": "blue", "data_type": "int16"},
            {"name": "sr_band3", "common_name": "green", "data_type": "int16"},
            {"name": "sr_band4", "common_name": "red", "data_type": "int16"},
            {"name": "sr_band5", "common_name": "rededge", "data_type": "int16"},
            {"name": "sr_band6", "common_name": "rededge", "data_type": "int16"},
            {"name": "sr_band7", "common_name": "rededge", "data_type": "int16"},
            {"name": "sr_band8", "common_name": "nir", "data_type": "int16"},
            {"name": "sr_band8a", "common_name": "nir08", "data_type": "int16"},
            {"name": "sr_band11", "common_name": "swir16", "data_type": "int16"},
            {"name": "sr_band12", "common_name": "swir22", "data_type": "int16"},
            {"name": "Fmask4", "common_name": "quality","data_type": "uint8"}
        ],
        "indexes": [
            {
                "name": "EVI",
                "common_name": "evi",
                "data_type": "int16",
                "metadata": {
                    "expression": {
                        "bands": [
                            "sr_band8",
                            "sr_band4",
                            "sr_band2"
                        ],
                        "value": "(10000. * 2.5 * (sr_band8 - sr_band4) / (sr_band8 + 6. * sr_band4 - 7.5 * sr_band2 + 10000.))"
                    }
                }
            },
            {
                "name": "NDVI",
                "common_name": "ndvi",
                "data_type": "int16",
                "metadata": {
                    "expression": {
                        "bands": [
                            "sr_band8",
                            "sr_band4"
                        ],
                        "value": "10000. * ((sr_band8 - sr_band4)/(sr_band8 + sr_band4))"
                    }
                }
            }
        ],
        "quality_band": "Fmask4",
        "description": "This data cube contains all available images from Sentinel-2, resampled to 10 meters of spatial resolution, reprojected, cropped and mosaicked to BDC_SM grid and time composed each 16 days using stack temporal composition function.",
        "parameters": {
            "mask": {
                "clear_data": [0, 1],
                "not_clear_data": [2, 3, 4],
                "nodata": 255,
                "saturated_data": []
            }
        }
    }'

In order to trigger a data cube, we are going to use a collection `S2_MSI_L2_SR_LASRC-1` made with Surface Reflectance using LaSRC 2.0 with cloud masking Fmask 4.2::

    # Using cube-builder command line
    cube-builder build S2_10_16D_LCF \
        --collections=S2_MSI_L2_SR_LASRC-1 \
        --tiles=089098 \
        --start=2019-01-01 \
        --end=2019-01-31


Creating data cube CBERS-4 AWFI
-------------------------------

In order to create data cube CBERS4 AWFI, use the following command to create data cube metadata:

.. code-block:: shell

    curl --location \
         --request POST '127.0.0.1:5000/cubes' \
         --header 'Content-Type: application/json' \
         --data-raw '
    {
        "datacube": "CB4",
        "grs": "BRAZIL_LG",
        "title": "CBERS-4 (AWFI) SR - Data Cube Stack 16 days - v001",
        "resolution": 64,
        "version": 1,
        "metadata": {
            "platform": {
              "code": "CBERS-4",
              "instruments": "AWFI"
            }
        },
        "temporal_composition": {
            "schema": "Cyclic",
            "step": 16,
            "unit": "day",
            "cycle": {
                "unit": "year",
                "step": 1
            }
        },
        "composite_function": "LCF",
        "bands_quicklook": [
            "sr_band12",
            "sr_band8a",
            "sr_band4"
        ],
        "bands": [
            {"name": "BAND13", "common_name": "blue", "data_type": "int16"},
            {"name": "BAND14", "common_name": "green", "data_type": "int16"},
            {"name": "BAND15", "common_name": "red", "data_type": "int16"},
            {"name": "BAND16", "common_name": "nir", "data_type": "int16"},
            {"name": "CMASK", "common_name": "quality","data_type": "uint8"}
        ],
        "indexes": [
            {
                "name": "EVI",
                "common_name": "evi",
                "data_type": "int16",
                "metadata": {
                    "expression": {
                        "bands": [
                            "BAND16",
                            "BAND15",
                            "BAND13"
                        ],
                        "value": "(10000. * 2.5 * (BAND16 - BAND15) / (BAND16 + 6. * BAND15 - 7.5 * BAND13 + 10000.))"
                    }
                }
            },
            {
                "name": "NDVI",
                "common_name": "ndvi",
                "data_type": "int16",
                "metadata": {
                    "expression": {
                        "bands": [
                            "BAND16",
                            "BAND15"
                        ],
                        "value": "10000. * ((BAND16 - BAND15)/(BAND16 + BAND15))"
                    }
                }
            }
        ],
        "quality_band": "CMASK",
        "description": "This data cube contains the all available images from CBERS-4/AWFI resampled to 64 meters of spatial resolution, reprojected and cropped to BDC_LG grid, composed each 16 days using the best pixel (Stack) composite function.",
        "parameters": {
            "mask": {
                "clear_data": [127],
                "not_clear_data": [255],
                "nodata": 0,
                "saturated_data": []
            }
        }
    }'

Trigger data cube generation with following command:

.. code-block:: shell

    # Using cube-builder command line
    cube-builder build CB4_64_16D_LCF \
        --collections=CBERS4_AWFI_L4_SR \
        --tiles=022024 \
        --start=2019-01-01 \
        --end=2019-01-31


Restarting or Reprocessing a Data Cube
--------------------------------------

When the ``Cube-Builder`` could not generate data cube for any unknown issue, you may restarting the entire process
with the same command you have dispatched::

    cube-builder build CB4_64_16D_LCF \
        --collections=CBERS4_AWFI_L4_SR \
        --tiles=022024 \
        --start=2019-01-01 \
        --end=2019-01-31

It will reuse most of files that were already processed, executing only the failed tasks. If you notice anything suspicious or want to re-create theses files again, use the option ``--force``::

    cube-builder build CB4_64_16D_LCF \
        --collections=CBERS4_AWFI_L4_SR \
        --tiles=022024 \
        --start=2019-01-01 \
        --end=2019-01-31 \
        --force


Data Cube Parameters
--------------------

The ``Cube-Builder`` supports a few parameters to be set during the data cube execution.

In order to check the parameters associated with data cube ``CB4_64_16D_STK-1``, use the command::

    cube-builder show-parameters CB4_64_16D_LCF-1


The following output represents all the parameters related with the given data cube::

    mask -> {'clear_data': [127], 'not_clear_data': [255], 'nodata': 0}
    quality_band -> CMASK
    stac_url -> https://brazildatacube.dpi.inpe.br/stac/
    token -> ChangeME


You can change any parameter with the command ``cube-builder configure`` with ``DataCubeName-Version``::

    cube-builder configure CB4_64_16D_LCF-1 --stac-url=AnySTAC

