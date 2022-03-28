..
    This file is part of Brazil Data Cube Builder.
    Copyright (C) 2019-2021 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


=======
Changes
=======


Version 0.6.4 (2022-03-29)
--------------------------

- Add tests for cube builder and workflow.
- Fix deprecated git protocol in pip for bdc-catalog package (`#211 <https://github.com/brazil-data-cube/cube-builder/issues/211>`_).
- Fix rasterio version for reading GeoTIFF or JP2 files (`#203 <https://github.com/brazil-data-cube/cube-builder/issues/203>`_).
- Review directory structure for data cubes (`#199 <https://github.com/brazil-data-cube/cube-builder/issues/199>`_).
- Set rio-cogeo version to 3 due GeoTransform bit precision casting (`#209 <https://github.com/brazil-data-cube/cube-builder/issues/209>`).
- Apply band valid ranges in cube identity (`#198 <https://github.com/brazil-data-cube/cube-builder/issues/203>`_).
- Fix bit precision error in grid generation (`#210 <https://github.com/brazil-data-cube/cube-builder/issues/203>`_).

Version 0.6.3 (2021-07-19)
--------------------------

- Improve the documentation how to deploy cube-builder on production environment using docker-compose (`#190 <https://github.com/brazil-data-cube/cube-builder/issues/190>`_).
- Remove the deprecated ModelSchema and use SQLAlchemyAutoSchema instead - marshmallow-sqlalchemy (`#192 <https://github.com/brazil-data-cube/cube-builder/issues/192>`_).


Version 0.6.2 (2021-04-20)
--------------------------

- Use masked array while composing data cubes (`#175 <https://github.com/brazil-data-cube/cube-builder/issues/175>`_).
- Fix ReadTheDocs build broken (https://cube-builder.readthedocs.io/en/latest/).
- Improve documentation how to setup Cube-Builder.



Version 0.6.1 (2021-04-12)
--------------------------

- Fix the generation of STACK product when ``saturated_data`` value is set to 1 (`#167 <https://github.com/brazil-data-cube/cube-builder/issues/167>`_).


Version 0.6.0 (2021-04-06)
--------------------------

- Change compression type to deflate (`#155 <https://github.com/brazil-data-cube/cube-builder/issues/155>`_)
- Add custom masking for data cubes (`#16 <https://github.com/brazil-data-cube/cube-builder/issues/16>`_)
- Add Histogram Equalization Matching support (`#138 <https://github.com/brazil-data-cube/cube-builder/issues/138>`_).
- Add routes to list pending and running tasks (`#103 <https://github.com/brazil-data-cube/cube-builder/issues/103>`_).
- Review the raster block size for data cubes (`#140 <https://github.com/brazil-data-cube/cube-builder/issues/140>`_).
- Add integration with Drone CI (`#149 <https://github.com/brazil-data-cube/cube-builder/pull/149>`_).
- Add integration with BDC-Auth (`#122 <https://github.com/brazil-data-cube/cube-builder/issues/122>`_).
- Improve data cube builder tasks distribution (`#47 <https://github.com/brazil-data-cube/cube-builder/issues/47>`_).
- Generate an empty raster file for period that does not contain any image (`#143 <https://github.com/brazil-data-cube/cube-builder/issues/143>`_).


Version 0.4.2 (2021-03-18)
--------------------------

- Fix stac resolver for CBERS or any external products (`#157 <https://github.com/brazil-data-cube/cube-builder/issues/157>`_)


Version 0.4.1 (2021-01-07)
--------------------------

- Add minimal way to secure cube-builder routes `#123 <https://github.com/brazil-data-cube/cube-builder/issues/123>`_.
- Allow to use custom prefix for each data cube item `#126 <https://github.com/brazil-data-cube/cube-builder/issues/126>`_
- Fix bug in reprocess tile cube through API  `#125 <https://github.com/brazil-data-cube/cube-builder/issues/125>`_.
- Fix bug in timeline period when creating a data cube - cyclic `#128 <https://github.com/brazil-data-cube/cube-builder/issues/128>`_.
- Fix bug when get cube metadata from a data cube that has not been executed `#130 <https://github.com/brazil-data-cube/cube-builder/issues/130>`_.
- Fix wrong cloud cover in data cubes `#131 <https://github.com/brazil-data-cube/cube-builder/issues/131>`_.


Version 0.4.0 (2020-12-03)
--------------------------

- Generate data cube vegetation index bands dynamically using `BDC-Catalog`. `#77 <https://github.com/brazil-data-cube/cube-builder/issues/77>`_.
- Prevent cube generation without quality band set. `#72 <https://github.com/brazil-data-cube/cube-builder/issues/72>`_.
- Add route to edit data cube metadata. `#113 <https://github.com/brazil-data-cube/cube-builder/issues/113>`_.
- Add support to generate data cube for only given composite function. `#12 <https://github.com/brazil-data-cube/cube-builder/issues/12>`_.
- Add support to generate data cubes from multiple sensors combined. `#9 <https://github.com/brazil-data-cube/cube-builder/issues/9>`_.
- Add option to specify any STAC URL. `#28 <https://github.com/brazil-data-cube/cube-builder/issues/28>`_.
- Add support to generate data cube using native collection grid (MGRS, WRS2, etc). `#104 <https://github.com/brazil-data-cube/cube-builder/pull/104>`_.
- Add support to reuse data cube Identity from another data cube. `#98 <https://github.com/brazil-data-cube/cube-builder/issues/98>`_.
- Integrate with BDC-Catalog `0.6.4 <https://github.com/brazil-data-cube/bdc-catalog/releases/tag/v0.6.4>`_.
- Fix bug in timeline for data cubes with cycle period. `#108 <https://github.com/brazil-data-cube/cube-builder/issues/108>`_.
- Fix dependency celery version. `#95 <https://github.com/brazil-data-cube/cube-builder/issues/95>`_.
- Fix mime type for thumbnails in assets. `#88 <https://github.com/brazil-data-cube/cube-builder/issues/88>`_.


Version 0.2.0 (2020-08-26)
--------------------------

- First experimental version.
- Create own Grid for the Data Cubes.
- Create spatial dimension for Data Cubes.
- Generate datacube from collections: Sentinel 2A/2B, Landsat-8 and CBERS-4 AWFI.
- Generate the products MEDIAN, STACK and IDENTITY data cubes.
- Documentation system based on Sphinx.
- Documentation integrated to ``Read the Docs``.
- Package support through Setuptools.
- Installation and deploy instructions.
- Schema versioning through Flask-Migrate.
- Source code versioning based on `Semantic Versioning 2.0.0 <https://semver.org/>`_.
- License: `MIT <https://raw.githubusercontent.com/brazil-data-cube/bdc-collection-builder/v0.2.0/LICENSE>`_.
