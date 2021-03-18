..
    This file is part of Brazil Data Cube Builder.
    Copyright (C) 2019-2020 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


=======
Changes
=======


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
