..
    This file is part of Cube Builder.
    Copyright (C) 2022 INPE.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.


Usage
=====

This section explains how to use the Cube Builder application to generate data cubes from Sentinel 2,
Landsat 8 and CBERS collections.


If you have not read yet how to install or deploy the system, please refer to :doc:`installation` or :doc:`deploy` documentation.
You may also read the `Brazil Data Cube Documentation <https://brazil-data-cube.github.io/>`_ for further details about Brazil Data Cube Project.


Temporal Compositing Functions
------------------------------

Before create any data cube, we strongly recommend you to read the `Temporal Compositing <https://brazil-data-cube.github.io/products/specifications/processing-flow.html#temporal-compositing>`_
to understand how these functions are defined in ``Cube-Builder``. In short, we have the following functions supported:

- ``Average`` (``AVG``): consist in the average of the observed values.
- ``Median`` (``MED``): consist in the median value of the observations.
- ``Least Cloud Cover First`` (``LCF``): consists in aggregating pixels from all images in the time interval
  according to each image quantity of valid pixels considering the image cloud cover efficacy.


Creating a Grid for the Data Cubes
----------------------------------


A Data Cube must have an associated grid as mentioned in `BDC Grid <https://brazil-data-cube.github.io/products/specifications/bdc-grid.html?highlight=grid>`_.
For this example, we will create 3 hierarchical grids for different data cubes:

- ``BRAZIL_SM``: used by collection ``Sentinel-2`` which has resolution ``10 meters``;
- ``BRAZIL_MD``: used by collection ``Landsat-8`` which has resolution ``30 meters``;
- ``BRAZIL_LG``: used by collection ``CBERS`` which has resolution ``64 meters`` (or  ``55 meters`` for ``CBERS4A``).

In order to do this, we must understand a few concepts:

- ``names``: A list consisting the Grid names to generate.
- ``tile_factor``: An ordered list of tile factor for the hierarchical grid scaling.
  We recommend you to use divisible values like `10`, `20` and `40`.
- ``shape``: A pivot shape to represent the tile factor. It usually consists in a desirable image shape for grid.
- ``meridian``: The central pivot for first grid. This will be reference/center value used to scale the other grids.


.. note::

    Keep in mind that the values ``shape`` and ``tile_factor`` are ``int`` values representing in meters.


.. note::

    Make sure that ``names`` and ``tile_factor`` **MUST HAVE** the same dimension. Each entry consists in
    a individual grid.


To create the new ``grids`` use the API Resource ``/create-grids`` as following::

    curl --location \
         --request POST '127.0.0.1:5000/create-grids' \
         --header 'Content-Type: application/json' \
         --data-raw '{
            "names": [
                "BRAZIL_SM",
                "BRAZIL_MD",
                "BRAZIL_LG"
            ],
            "description": "Brazil Grids - Albers Equal Area",
            "projection": "aea",
            "meridian": -54,
            "tile_factor": [
                [10, 10],
                [20, 20],
                [40, 40]
            ],
            "shape": [10560, 10560],
            "bbox": [-73.98318215899995, -33.75117799399993, -28.847770352999916, 5.269580833000035],
            "srid": 100001
         }'

The response will have status code ``201`` and the body::

    "Grids ['BRAZIL_SM', 'BRAZIL_MD', 'BRAZIL_LG'] created with successfully"


.. note::

    You may create non-hierarchical Grid, just specify your own arguments like shape, bbox, shape and use single name
    and tile_factor::

        curl --location \
             --request POST '127.0.0.1:5000/create-grids' \
             --header 'Content-Type: application/json' \
             --data-raw '{
                "names": [
                    "BRAZIL_SM"
                ],
                "description": "Brazil Grids - Albers Equal Area",
                "projection": "aea",
                "meridian": -54,
                "tile_factor": [
                    [10, 10]
                ],
                "shape": [10560, 10560],
                "bbox": [-73.98318215899995, -33.75117799399993, -28.847770352999916, 5.269580833000035],
                "srid": 100001
             }'


Creating data cube Landsat-8
----------------------------

In order to create data cube ``Landsat-8`` monthly using the composite function ``Least Cloud Cover First`` (`LC8-1M`), use the following command to create data cube metadata::

    curl --location \
         --request POST '127.0.0.1:5000/cubes' \
         --header 'Content-Type: application/json' \
         --data-raw '
    {
        "datacube": "LC8-1M",
        "datacube_identity": "LC8",
        "grs": "BRAZIL_MD",
        "title": "Landsat-8 (OLI) Cube Monthly - v001",
        "resolution": 30,
        "version": 1,
        "metadata": {
            "license": "proprietary",
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
            "sr_band4",
            "sr_band3",
            "sr_band2"
        ],
        "bands": [
            {"name": "sr_band1", "common_name": "coastal", "data_type": "int16", "nodata": -9999},
            {"name": "sr_band2", "common_name": "blue", "data_type": "int16", "nodata": -9999},
            {"name": "sr_band3", "common_name": "green", "data_type": "int16", "nodata": -9999},
            {"name": "sr_band4", "common_name": "red", "data_type": "int16", "nodata": -9999},
            {"name": "sr_band5", "common_name": "nir", "data_type": "int16", "nodata": -9999},
            {"name": "sr_band6", "common_name": "swir1", "data_type": "int16", "nodata": -9999},
            {"name": "sr_band7", "common_name": "swir2", "data_type": "int16", "nodata": -9999},
            {"name": "Fmask4", "common_name": "quality", "data_type": "uint8", "nodata": 255}
        ],
        "indexes": [
            {
                "name": "EVI",
                "common_name": "evi",
                "data_type": "int16",
                "nodata": -9999,
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
                "nodata": -9999,
                "metadata": {
                    "expression": {
                        "bands": ["sr_band5", "sr_band4"],
                        "value": "10000. * ((sr_band5 - sr_band4)/(sr_band5 + sr_band4))"
                    }
                }
            }
        ],
        "quality_band": "Fmask4",
        "description": "This datacube contains the all available images from Landsat-8, with 30 meters of spatial resolution, reprojected and cropped to BDC_MD grid, composed each 16 days using the best pixel (LCF) composite function.",
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
    The ``Clear Data`` pixels are considered to identify the ``Best Pixel`` (LCF) and it is count on the ``Clear Observation Band`` (``ClearOb``).

In order to trigger a data cube, we are going to use a collection `LC8_SR-1` made with Surface Reflectance using LaSRC 2.0 with cloud masking Fmask 4.2.
In this example, we are going to use the official `Brazil Data Cube STAC <https://brazildatacube.dpi.inpe.br/stac/>`_. To do so, you will need to have an account in
Brazil Data Cube environment. If you don't have any account, please, refer to `Brazil Data Cube Explorer <https://brazil-data-cube.github.io/applications/dc_explorer/token-module.html>`_.

Once the data cube definition is created, you can trigger a data cube using the following command::

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder build LC8-1M-1 \
        --stac-url https://brazildatacube.dpi.inpe.br/stac/ \
        --collections=LC8_SR-1 \
        --tiles=011009 \
        --start=2019-01-01 \
        --end=2019-01-31 \
        --token <USER_BDC_TOKEN>

.. note::

    If you would like to trigger data cube generation using ``API call`` instead ``commandline`` use as following::

        # Using curl (Make sure to execute cube-builder run)
        curl --location \
             --request POST '127.0.0.1:5000/start' \
             --header 'Content-Type: application/json' \
             --data-raw '{
                "stac_url": "https://brazildatacube.dpi.inpe.br/stac/",
                "token": "<USER_BDC_TOKEN>",
                "datacube": "LC8-1M",
                "collections": ["LC8_SR-1"],
                "tiles": ["011009"],
                "start_date": "2019-01-01",
                "end_date": "2019-01-31"
             }'


.. note::

    The command line ``cube-builder build`` has few optional parameters such
    ``bands``, which defines bands to generate data cube.

    You may also pass ``--stac-url=URL_TO_STAC`` (command line) or ``"stac_url": "URL_TO_STAC"`` (API only)
    if you would like to generate data cube using a different STAC provider. Remember that the ``--collection`` must exists.


.. _create_sentinel:

Creating data cube Sentinel 2
-----------------------------

In order to create data cube Sentinel 2, use the following command to create data cube metadata:

.. code-block:: shell

    curl --location \
         --request POST '127.0.0.1:5000/cubes' \
         --header 'Content-Type: application/json' \
         --data-raw '
    {
        "datacube": "S2-16D",
        "datacube_identity": "S2",
        "grs": "BRAZIL_SM",
        "title": "Sentinel-2 SR - Cube LCF 16 days -v001",
        "resolution": 10,
        "version": 1,
        "metadata": {
            "license": "proprietary",
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
            "B04",
            "B03",
            "B02"
        ],
        "bands": [
            {"name": "B01", "common_name": "coastal", "data_type": "int16", "nodata": 0},
            {"name": "B02", "common_name": "blue", "data_type": "int16", "nodata": 0},
            {"name": "B03", "common_name": "green", "data_type": "int16", "nodata": 0},
            {"name": "B04", "common_name": "red", "data_type": "int16", "nodata": 0},
            {"name": "B05", "common_name": "rededge", "data_type": "int16", "nodata": 0},
            {"name": "B06", "common_name": "rededge", "data_type": "int16", "nodata": 0},
            {"name": "B07", "common_name": "rededge", "data_type": "int16", "nodata": 0},
            {"name": "B08", "common_name": "nir", "data_type": "int16", "nodata": 0},
            {"name": "B8A", "common_name": "nir08", "data_type": "int16", "nodata": 0},
            {"name": "B11", "common_name": "swir16", "data_type": "int16", "nodata": 0},
            {"name": "B12", "common_name": "swir22", "data_type": "int16", "nodata": 0},
            {"name": "SCL", "common_name": "quality","data_type": "uint8", "nodata": 0}
        ],
        "indexes": [
            {
                "name": "EVI",
                "common_name": "evi",
                "data_type": "int16",
                "nodata": -9999,
                "metadata": {
                    "expression": {
                        "bands": [
                            "B8A",
                            "B04",
                            "B02"
                        ],
                        "value": "(10000. * 2.5 * (B8A - B04) / (B8A + 6. * B04 - 7.5 * B02 + 10000.))"
                    }
                }
            },
            {
                "name": "NDVI",
                "common_name": "ndvi",
                "data_type": "int16",
                "nodata": -9999,
                "metadata": {
                    "expression": {
                        "bands": [
                            "B8A",
                            "B04"
                        ],
                        "value": "10000. * ((B8A - B04)/(B8A + B04))"
                    }
                }
            }
        ],
        "quality_band": "SCL",
        "description": "This data cube contains all available images from Sentinel-2, resampled to 10 meters of spatial resolution, reprojected, cropped and mosaicked to BDC_SM grid and time composed each 16 days using LCF temporal composition function.",
        "parameters": {
            "mask": {
                "clear_data": [4, 5, 6],
                "not_clear_data": [2, 3, 7, 8, 9, 10, 11],
                "nodata": 0,
                "saturated_data": [1]
            }
        }
    }'

In order to trigger a data cube, we are going to use a collection `S2-16-1` made with Surface Reflectance using Sen2Cor::

    # Using cube-builder command line
    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder build S2-16D-1 \
        --stac-url https://brazildatacube.dpi.inpe.br/stac/ \
        --collections=S2_L2A-1 \
        --tiles=017019 \
        --start=2019-01-01 \
        --end=2019-01-31 \
        --token <USER_BDC_TOKEN>


Creating data cube CBERS-4 AWFI
-------------------------------

In order to create data cube CBERS4 AWFI, use the following command to create data cube metadata:

.. code-block:: shell

    curl --location \
         --request POST '127.0.0.1:5000/cubes' \
         --header 'Content-Type: application/json' \
         --data-raw '
    {
        "datacube": "CB4-16D",
        "datacube_identity": "CB4",
        "grs": "BRAZIL_LG",
        "title": "CBERS-4 (AWFI) SR - Data Cube LCF 16 days - v001",
        "resolution": 64,
        "version": 1,
        "metadata": {
            "license": "cc-by-sa-3.0",
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
            "BAND15",
            "BAND14",
            "BAND13"
        ],
        "bands": [
            {"name": "BAND13", "common_name": "blue", "data_type": "int16", "nodata": -9999},
            {"name": "BAND14", "common_name": "green", "data_type": "int16", "nodata": -9999},
            {"name": "BAND15", "common_name": "red", "data_type": "int16", "nodata": -9999},
            {"name": "BAND16", "common_name": "nir", "data_type": "int16", "nodata": -9999},
            {"name": "CMASK", "common_name": "quality","data_type": "uint8", "nodata": 0}
        ],
        "indexes": [
            {
                "name": "EVI",
                "common_name": "evi",
                "data_type": "int16",
                "nodata": -9999,
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
                "nodata": -9999,
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
        "description": "This data cube contains the all available images from CBERS-4/AWFI resampled to 64 meters of spatial resolution, reprojected and cropped to BDC_LG grid, composed each 16 days using the best pixel (LCF) composite function.",
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
    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder build CB4-16D-1 \
        --stac-url https://brazildatacube.dpi.inpe.br/stac/ \
        --collections=CBERS4_AWFI_L4_SR \
        --tiles=005004 \
        --start=2019-01-01 \
        --end=2019-01-31 \
        --token <USER_BDC_TOKEN>


Restarting or Reprocessing a Data Cube
--------------------------------------

When the ``Cube-Builder`` could not generate data cube for any unknown issue, you may restarting the entire process
with the same command you have dispatched::

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder build CB4-16D-1 \
        --stac-url https://brazildatacube.dpi.inpe.br/stac/ \
        --collections=CBERS4_AWFI_L4_SR \
        --tiles=005004 \
        --start=2019-01-01 \
        --end=2019-01-31 \
        --token <USER_BDC_TOKEN>

It will reuse most of files that were already processed, executing only the failed tasks. If you notice anything suspicious or want to re-create theses files again, use the option ``--force``::

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder build CB4-16D-1 \
        --stac-url https://brazildatacube.dpi.inpe.br/stac/ \
        --collections=CBERS4_AWFI_L4_SR \
        --tiles=005004 \
        --start=2019-01-01 \
        --end=2019-01-31 \
        --token <USER_BDC_TOKEN> \
        --force


Data Cube Parameters
--------------------

The ``Cube-Builder`` supports a few parameters to be set during the data cube execution.

In order to check the parameters associated with data cube ``CB4-16D-1``, use the command::

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder show-parameters CB4-16D-1


The following output represents all the parameters related with the given data cube::

    mask -> {'clear_data': [127], 'not_clear_data': [255], 'nodata': 0}
    quality_band -> CMASK
    stac_url -> https://brazildatacube.dpi.inpe.br/stac/
    token -> ChangeME


You can change any parameter with the command ``cube-builder configure`` with ``DataCubeName-Version``::

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder configure CB4-16D-1 --stac-url=AnySTAC


.. note::

    Once parameter is set, it only be affected in the new execution.
    Be aware of what you are changing to do not affect the integrity of data cube.
    For example, changing the masking ``clear_data`` when there is a already area generated.
    Make sure to re-generate all the periods and tiles again.



Advanced User Guide
-------------------

Generate data cubes from local dir
++++++++++++++++++++++++++++++++++

.. versionadded:: 1.0.0

.. note::

    To proceed this step, you will need to have a set of files in disk.
    We will not provide this files since its just a briefing of this feature. You may consider
    to have own files individually.


With latest change of ``Cube-Builder`` (1.0), the user can generate data cubes using local directories containing
images. This feature is useful to generate data cubes when the user has a bunch of image files locally and would like
to apply temporal composition function over these files. In this case, a  ``STAC Server`` is not required.
This feature can be achieved using parameters ``--local DIRECTORY`` and ``--format PATH_TO_FORMAT.json``.
It follows the signature of `GDALCubes Formats <https://github.com/appelmar/gdalcubes/tree/master/formats>`_ to read
directories.
Essentially, a format contains the following properties:

- ``images`` (REQUIRED): Object context representing how to seek for any image in disk.

    - ``pattern`` (REQUIRED)
- ``datetime`` (REQUIRED): Object context describing how to identify data times from any directory path or file path.

    - ``pattern`` (REQUIRED): A regex expression describing how to match datetime.
    - ``format`` (REQUIRED): ISO Format to get data time from `str`.

- ``bands`` (REQUIRED): The data set bands that will be captured while recurring disk. You can also add extra fields to increment metadata of band. The following internal props are required:

    - ``pattern``: Regex pattern to identify band in disk.
    - ``nodata``: No data value for band.
- ``tags`` (OPTIONAL): List of keywords describing the given format.
- ``description`` (OPTIONAL): A detailed multi-line description to fully explain the format.

You can check a minimal example in ``examples/formats/bdc-sentinel-2-l2a-cogs.json``, which offers support to
locate ``Sentinel-2`` Cloud Optimized GeoTIFF files. You may also take a look in `GDALCubes Formats <https://github.com/appelmar/gdalcubes/tree/master/formats>`_
for others formats.

For this example, lets create a simple sentinel-2 data cube called ``S2-LOCAL-16D``. The signature is similar from
:ref:`create_sentinel`. We just need to change the cube parameters to something like::

        ...
        "parameters": {
            "mask": {
                "clear_data": [4, 5, 6],
                "not_clear_data": [2, 3, 7, 8, 9, 10, 11],
                "nodata": 0,
                "saturated_data": [1]
            },
            "local": "/path/to/local/files",
            "recursive": true,
            "format": "examples/formats/bdc-sentinel-2-l2a-cogs.json",
            "pattern": ".tif"
        }

So you can create a data cube with command::

    curl --location \
         --request POST '127.0.0.1:5000/cubes' \
         --header 'Content-Type: application/json' \
         --data-raw '
    {
        "datacube": "S2-LOCAL-16D",
        "datacube_identity": "S2-LOCAL",
        "grs": "BRAZIL_SM",
        "title": "Sentinel-2 SR - Cube LCF 16 days -v001",
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
            "B04",
            "B03",
            "B02"
        ],
        "bands": [
            {"name": "B01", "common_name": "coastal", "data_type": "int16", "nodata": 0},
            {"name": "B02", "common_name": "blue", "data_type": "int16", "nodata": 0},
            {"name": "B03", "common_name": "green", "data_type": "int16", "nodata": 0},
            {"name": "B04", "common_name": "red", "data_type": "int16", "nodata": 0},
            {"name": "B05", "common_name": "rededge", "data_type": "int16", "nodata": 0},
            {"name": "B06", "common_name": "rededge", "data_type": "int16", "nodata": 0},
            {"name": "B07", "common_name": "rededge", "data_type": "int16", "nodata": 0},
            {"name": "B08", "common_name": "nir", "data_type": "int16", "nodata": 0},
            {"name": "B8A", "common_name": "nir08", "data_type": "int16", "nodata": 0},
            {"name": "B11", "common_name": "swir16", "data_type": "int16", "nodata": 0},
            {"name": "B12", "common_name": "swir22", "data_type": "int16", "nodata": 0},
            {"name": "SCL", "common_name": "quality","data_type": "uint8", "nodata": 0}
        ],
        "indexes": [
            {
                "name": "EVI",
                "common_name": "evi",
                "data_type": "int16",
                "nodata": -9999,
                "metadata": {
                    "expression": {
                        "bands": [
                            "B8A",
                            "B04",
                            "B02"
                        ],
                        "value": "(10000. * 2.5 * (B8A - B04) / (B8A + 6. * B04 - 7.5 * B02 + 10000.))"
                    }
                }
            },
            {
                "name": "NDVI",
                "common_name": "ndvi",
                "data_type": "int16",
                "nodata": -9999,
                "metadata": {
                    "expression": {
                        "bands": [
                            "B8A",
                            "B04"
                        ],
                        "value": "10000. * ((B8A - B04)/(B8A + B04))"
                    }
                }
            }
        ],
        "quality_band": "SCL",
        "description": "This data cube contains all available images from Sentinel-2, resampled to 10 meters of spatial resolution, reprojected, cropped and mosaicked to BDC_SM grid and time composed each 16 days using LCF temporal composition function.",
        "parameters": {
            "mask": {
                "clear_data": [4, 5, 6],
                "not_clear_data": [2, 3, 7, 8, 9, 10, 11],
                "nodata": 0,
                "saturated_data": [1]
            },
            "local": "/path/to/local/files",
            "recursive": true,
            "format": "examples/formats/bdc-sentinel-2-l2a-cogs.json",
            "pattern": ".tif"
        }
    }'

After cube definition created, you can just use the command line ``cube-builder build-local``::

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder build-local S2-LOCAL-16D-1 \
        --tiles 003011 \
        --start-date 2021-08-29 \
        --end-date 2021-09-13 \
        --directory /path/to/local/files \
        --format examples/formats/bdc-sentinel-2-l2a-cogs.json


.. note::

    This example just illustrate how to trigger the data cube using local directory. You may need to
    change these values like ``directory``, ``format``, ``start-date``, ``end-date`` and ``tiles``.

    Right now, it only supports using ``--tiles`` as parameter. It will be replaced in the next release
    to support any ``Region of Interest (ROI)`` or shapefile.
