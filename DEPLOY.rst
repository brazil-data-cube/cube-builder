..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2021 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Deploying
=========

The ``docker-compose.yml`` in the root of the source tree can be used to run cube-builder as a multi-container application.


This section explains how to get the cube-builder service up and running with Docker and Docker Compose.
If you do not have Docker installed, take a look at `this tutorial on how to install it in your system <https://docs.docker.com/install/>`_.
See also the `tutorial on how to install Docker Compose <https://docs.docker.com/compose/install/>`_.


Compatibility
-------------

Before deploy/install ``Cube-Builder``, please, take a look into compatibility table:

+--------------+-------------+
| Cube-Builder | BDC-Catalog |
+==============+=============+
| 0.8.x        | 0.8.2       |
+--------------+-------------+
| 0.4.x, 0.6.x | 0.8.1       |
+--------------+-------------+
| 0.2.x        | 0.2.x       |
+--------------+-------------+


Configuration
-------------

Before proceed to the ``DEPLOY`` step, we have prepared a minimal ``docker-compose.yml`` file
with config to launch ``Cube-Builder``.
By default, it will generate data cubes in directories ``./volumes/data`` and ``./volumes/workdir``, respectively.
You may set a different location editing the ``docker-compose.yml`` file. Please refer to the page :doc:`configuration`
for further details.

.. note::

    Take a look into ``docker-compose.yml`` the variables ``DATA_DIR`` and ``WORK_DIR``
    and make sure you have enough space in disk for data cubes.


Running the Docker Containers
-----------------------------

.. note::

    Make sure you have a machine with at least the following requirements:

        - 4 vCPU or more
        - 8 GB RAM
        - 40 GB free space


.. note::

    If you do not have a PostgreSQL instance with the Brazil Data Cube data model up and running, you will need to prepare one before following the rest of this documentation.


    In order to launch a PostgreSQL container, you can rely on the docker-compose service file. The following command will start a new container with PostgreSQL:

    .. code-block:: shell

        $ docker-compose up -d postgres


    Please check the `docker-compose.yml` file. It will expose the following database parameters available in docker context environment::

        SQLALCHEMY_DATABASE_URI=postgresql://postgres:postgres@cube-builder-pg:5432/bdc

    After launching the container, please, refer to the section "Prepare the Database System" in the `INSTALL.rst <INSTALL.rst>`_ documentation. This will guide you in the preparation of the PostgreSQL setup.

    For docker environment, you may run the database step using the following command::

        docker-compose run --name prepare-db --rm cube-builder ./deploy/configure-db.sh

    If the above command runs successfully, you will be able to see the database creation as following::

        Starting cube-builder-rabbitmq ... done
        Creating cube-builder_cube-builder_run ... done
        Creating database postgresql://postgres:postgres@cube-builder-pg:5432/bdc...
        Database created!
        Creating namespace lccs...
        Creating namespace bdc...
        Creating namespace cube_builder...
        Namespaces created!
        Creating extension postgis...
        Extension created!
        Creating database schema...
          [####################################]  100%
        Database schema created!
        Registering triggers from "bdc_catalog.triggers"
                -> /usr/local/lib/python3.8/site-packages/bdc_catalog/triggers/timeline.sql
                -> /usr/local/lib/python3.8/site-packages/bdc_catalog/triggers/band_metadata_expression.sql
                -> /usr/local/lib/python3.8/site-packages/bdc_catalog/triggers/collection_statistics.sql
        Triggers from "bdc_catalog.triggers" registered



Use the following command in order to launch all the containers needed to run cube-builder [#f1]_:

.. code-block:: shell

    $ docker-compose up -d


If the above command runs successfully, you will be able to list the launched containers:

.. code-block:: shell

    $ docker container ls

    CONTAINER ID        IMAGE                                                      COMMAND                  CREATED             STATUS              PORTS                    NAMES
    a3bb86d2df56        rabbitmq:3-management                                      "docker-entrypoint.s…"   3 minutes ago       Up 2 minutes        4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp   cube-builder-rabbitmq
    e3862ab6e756        registry.dpi.inpe.br/brazil-data-cube/cube-builder:latest  "bash -c 'cube-build…"   2 minutes ago       Up 2 minutes        0.0.0.0:5001->5000/tcp   cube-builder-api
    13caa0f27030        registry.dpi.inpe.br/brazil-data-cube/cube-builder:latest  "cube-builder worker…"   2 minutes ago       Up 2 minutes                                 cube-builder-worker


.. note::

    Refer to the page :doc:`usage` documentation in order to use the cube builder services.


.. rubric:: Footnotes

.. [#f1]

    | By default, the docker compose will try to build a new Docker image
    | If you have account in the BDC registry, you may use as following:
    | ``$ docker login registry.dpi.inpe.br``
