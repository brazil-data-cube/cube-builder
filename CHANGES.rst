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


=======
Changes
=======


Version 1.0.1 (2023-06-27)
--------------------------

- Fix bug related session close in SQLAlchemy & Celery (`#264 <https://github.com/brazil-data-cube/cube-builder/issues/264>`_)
- Fix bug related docs and deploy database models in `deploy/configure-db.sh` (`#263 <https://github.com/brazil-data-cube/cube-builder/issues/263>`_)
- Improve route to list cubes and add support to filter by collection type (`#262 <https://github.com/brazil-data-cube/cube-builder/issues/262>`_)
- Add initial support for Python 3.11+
- Review code organization for datasets
- Remove warnings related Python 3.11 and libraries
- Add support to apply Sentinel-2 baseline offsetting (`#266 <https://github.com/brazil-data-cube/cube-builder/issues/266>`_)


Version 1.0.0 (2023-03-10)
--------------------------

- Add support to generate data cubes for EO Landsat combined sensors (`#172 <https://github.com/brazil-data-cube/cube-builder/issues/172>`_)
- Improve generation of Data Cube preview images.
- Add support for interoperability of STAC clients: 0.9.x and 1.0
- Add integration with BDC-Catalog 1.0 `#233 <https://github.com/brazil-data-cube/cube-builder/issues/233>`_.
- Add support to generate datacube from Sentinel-2 Zip (experimental) `#222 <https://github.com/brazil-data-cube/cube-builder/issues/222>`_.
- Add support to generate data cube from local directories (`#25 <https://github.com/brazil-data-cube/cube-builder/issues/25>`_)
- Upgrade celery to 5.x (`#251 <https://github.com/brazil-data-cube/cube-builder/issues/251>`_).
- Deprecate internal parameters related reuse cube
- Fix bug related dependencies: "jinja2" (`#258 <https://github.com/brazil-data-cube/cube-builder/issues/251>`_).
- Report warning message for invalid tile or expired token while generating datacubes (`#245 <https://github.com/brazil-data-cube/cube-builder/issues/245>`_).


Version 0.8.5 (2023-03-08)
--------------------------

- Fix integration with STAC v1 and STAC Legacy versions
- Add notice in INSTALL for compatibility with package and "setuptools<67"


Version 1.0.0a2 (2023-01-30)
----------------------------

- Improve docs setup
- Add integration with 0.8.4 and generate data cubes for EO Landsat combined sensors (`#172 <https://github.com/brazil-data-cube/cube-builder/issues/172>`_)
- Add support for interoperability of STAC clients: 0.9.x and 1.0


Version 0.8.4 (2023-01-23)
--------------------------

- Add support to generate data cube from Landsat Collection 2 (`#172 <https://github.com/brazil-data-cube/cube-builder/issues/172>`_)
- Add support to combine Landsat Collection 2 sensors (L5/L7, L7/L8, L7/L8/L9) using single collection (`#172 <https://github.com/brazil-data-cube/cube-builder/issues/172>`_)
- Review API error when no parameter is set.
- Review unittests for cube creation and code coverage.


Version 1.0.0a1 (2022-10-20)
----------------------------

- Add integration with BDC-Catalog 1.0 `#233 <https://github.com/brazil-data-cube/cube-builder/issues/233>`_.
- Add support to generate datacube from Sentinel-2 Zip (experimental) `#222 <https://github.com/brazil-data-cube/cube-builder/issues/222>`_.
- Improve docs setup
- Add support to generate data cube from local directories (`#25 <https://github.com/brazil-data-cube/cube-builder/issues/25>`_)


Version 0.8.3 (2022-10-03)
--------------------------

- Add support to customize data cube path and data cube item (`#236 <https://github.com/brazil-data-cube/cube-builder/issues/236>`_)
- Review docs related with new path format cubes


Version 0.8.2 (2022-09-21)
--------------------------

- Change LICENSE to GPL v3 and headers source code


Version 0.8.1 (2022-09-14)
--------------------------

- Fix search images using temporal dimension `#221 <https://github.com/brazil-data-cube/cube-builder/issues/221>`_.
- Add way to retrieve total items per tile
- Improve Usage/Setup documentation `#164 <https://github.com/brazil-data-cube/cube-builder/issues/164>`_.


Version 0.8.0 (2022-05-05)
--------------------------

- Add tests for cube builder and workflow.
- Fix deprecated git protocol in pip for bdc-catalog package (`#211 <https://github.com/brazil-data-cube/cube-builder/issues/211>`_).
- Fix rasterio version for reading GeoTIFF or JP2 files (`#203 <https://github.com/brazil-data-cube/cube-builder/issues/203>`_).
- Review directory structure for data cubes (`#199 <https://github.com/brazil-data-cube/cube-builder/issues/199>`_).
- Set rio-cogeo version to 3 due GeoTransform bit precision casting (`#209 <https://github.com/brazil-data-cube/cube-builder/issues/209>`).
- Apply band valid ranges in cube identity (`#198 <https://github.com/brazil-data-cube/cube-builder/issues/203>`_).
- Fix bit precision error in grid generation (`#210 <https://github.com/brazil-data-cube/cube-builder/issues/203>`_).
- Add support to generate data cube Identity without Mask band (`#142 <https://github.com/brazil-data-cube/cube-builder/issues/142>`_).
- Rename Stack composite function to Least CC First (LCF) (`#213 <https://github.com/brazil-data-cube/cube-builder/issues/213>`_).
- Add support to retrieve GRID tiles GeoJSON through API (`#215 <https://github.com/brazil-data-cube/cube-builder/issues/215>`_)
- Add flag "data_dir" to customize data cube location (`#205 <https://github.com/brazil-data-cube/cube-builder/issues/205>`_)


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
