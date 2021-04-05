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


Configuration
-------------

docker-compose.yml
~~~~~~~~~~~~~~~~~~

Open and edit **docker-compose.yml** with the following variables:

1. **DATA_DIR** - Path to store collections.
2. **SQLALCHEMY_DATABASE_URI** - Database URI.
3. **RABBIT_MQ_URL** - URI to connect on RabbitMQ protocol.


Running the Docker Containers
-----------------------------

.. note::

    If you do not have a PostgreSQL instance with the Brazil Data Cube data model up and running, you will need to prepare one before following the rest of this documentation.


    In order to launch a PostgreSQL container, you can rely on the docker-compose service file. The following command will start a new container with PostgreSQL:

    .. code-block:: shell

        $ docker-compose up -d postgres


    After launching the container, please, refer to the section "Prepare the Database System" in the `INSTALL.rst <./INSTALL.rst>`_ documentation. This will guide you in the preparation of the PostgreSQL setup.


Use the following command in order to launch all the containers needed to run cube-builder [#f1]_:

.. code-block:: shell

    $ docker-compose up -d


If the above command runs successfully, you will be able to list the launched containers:

.. code-block:: shell

    $ docker container ls

    CONTAINER ID        IMAGE                                                    COMMAND                  CREATED             STATUS              PORTS                    NAMES
    a3bb86d2df56        rabbitmq:3-management                                    "docker-entrypoint.s…"   3 minutes ago       Up 2 minutes        4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp   cube-builder-rabbitmq
    e3862ab6e756        registry.dpi.inpe.br/brazildatacube/cube-builder:latest  "bash -c 'cube-build…"   2 minutes ago       Up 2 minutes        0.0.0.0:5001->5000/tcp   cube-builder-api
    13caa0f27030        registry.dpi.inpe.br/brazildatacube/cube-builder:latest  "cube-builder worker…"   2 minutes ago       Up 2 minutes                                 cube-builder-worker


.. note::

    Refer to the `USING.rst <./USING.rst>`_ documentation in order to use the cube builder services.


.. rubric:: Footnotes

.. [#f1]

    | For now you will need to login into the BDC registry:
    | ``$ docker login registry.dpi.inpe.br``
    |
    | In the next releases we will get ride of this internal registry.