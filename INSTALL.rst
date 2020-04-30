..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2020 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Installation
============

The ``cube-buider`` depends essentially on:

- `Python Client Library for STAC (stac.py) <https://github.com/brazil-data-cube/stac.py>`_

- `Flask <https://palletsprojects.com/p/flask/>`_

- `Celery <http://www.celeryproject.org/>`_

- `rasterio <https://rasterio.readthedocs.io/en/latest/>`_

- `GDAL <https://gdal.org/>`_ ``(Version 2+ or 3+)``

- `NumPy <https://numpy.org/>`_

- `scikit-image <https://scikit-image.org/>`_

- `RabbitMQ <https://www.rabbitmq.com/>`_

- `marshmallow-SQLAlchemy <https://marshmallow-sqlalchemy.readthedocs.io/en/latest/>`_

- `Brazil Data Cube DataBase Module <https://github.com/brazil-data-cube/bdc-db.git>`_

- `Brazil Data Cube Core Module <https://github.com/brazil-data-cube/bdc-core.git>`_


Development Installation
------------------------

Clone the software repository:

.. code-block:: shell

    $ git clone https://github.com/brazil-data-cube/cube-builder.git


Go to the source code folder:

.. code-block:: shell

    $ cd cube-builder


Install in development mode:

.. code-block:: shell

    $ pip3 install -e .[all]

.. note::

    | If you have problems during the GDAL Python package installation, please, make sure to have the GDAL library support installed in your system with its command line tools.
    |
    | You can check the GDAL version with:
    | ``$ gdal-config --version``.
    |
    | Then, if you want to install a specific version (example: 2.4.2), try:
    | ``$ pip install "gdal==2.4.2"``
    |
    | If you still having problems with GDAL installation, you can generate a log in order to check what is happening with your installation. Use the following ``pip`` command:
    | ``$ pip install --verbose --log my.log "gdal==2.4.2"``
    |
    | For more information, see [#f1]_ e [#f2]_.


Running in Development Mode
---------------------------


Launch the RabbitMQ Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will need an instance of RabbitMQ up and running in order to launch the cube-builder celery workers.
We have prepared in the ``cube-builder`` repository a configuration for RabbitMQ container on docker-compose.
Please, follow the steps below:

.. code-block:: shell

        docker-compose up -d mq


After that command, check which port was binded from the host to the container:

.. code-block:: shell

        $ docker container ls

        CONTAINER ID   IMAGE                  COMMAND                  CREATED         STATUS         PORTS                    NAMES
        a3bb86d2df56   rabbitmq:3-management  "docker-entrypoint.s…"   3 minutes ago   Up 3 minutes   4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp   cube-builder-rabbitmq


.. note::

        Note that in the above output the RabbitMQ service is attached to the ports ``5672`` for socket client and
        ``15672`` the RabbitMQ User Interface. You can check `<http://127.0.0.1:15672>`_. The default credentials are ``guest`` and ``guest`` for
        user and password respectively.


Prepare the Database System
~~~~~~~~~~~~~~~~~~~~~~~~~~~


The ``cube-builder`` uses `bdc-db <https://github.com/brazil-data-cube/bdc-db/>`_ as database definition to store data cube metadata.

In order to prepare a Brazil Data Cube database model, you must clone the ``bdc-db`` and run the migrations:

.. code-block:: shell

    git clone https://github.com/brazil-data-cube/bdc-db.git /tmp/bdc-db
    (
        cd /tmp/bdc-db
        SQLALCHEMY_DATABASE_URI="postgresql://postgres:bdc-collection-builder2019@localhost:5432/cube-builder" \
        bdc-db db create-db
        SQLALCHEMY_DATABASE_URI="postgresql://postgres:bdc-collection-builder2019@localhost:5432/cube-builder" \
        bdc-db db upgrade
    )

After that, you can initialize Cube Builder migrations with the following commands:


.. code-block:: shell

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:bdc-collection-builder2019@localhost:5432/cube-builder" \
    cube-builder db create-db # Create database and schema

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:bdc-collection-builder2019@localhost:5432/cube-builder" \
    cube-builder db upgrade # Up migrations

    # Load default functions for cube-builder
    SQLALCHEMY_DATABASE_URI="postgresql://postgres:bdc-collection-builder2019@localhost:5432/cube-builder" \
    cube-builder load-data


Launch the cube-builder service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the source code folder, enter the following command:

.. code-block:: shell

        $ FLASK_ENV="development" \
          DATA_DIR="/data" \
          SQLALCHEMY_DATABASE_URI="postgresql://postgres:password@host:port/bdc" \
          cube-builder run


You may need to replace the definition of some environment variables:

- ``FLASK_ENV="development"``: used to tell Flask to run in ``Debug`` mode.

- ``DATA_DIR="/data"``: set path to store data cubes

- ``SQLALCHEMY_DATABASE_URI="postgresql://postgres:password@host:port/bdc"``: set the database URI connection for PostgreSQL.


The above command should output some messages in the console as showed below:

.. code-block:: shell

    * Environment: development
    * Debug mode: on
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    * Restarting with stat
    * Debugger is active!
    * Debugger PIN: 319-592-254


Launch the cube-builder worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enter the following command to start cube-builder worker:

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
    You can limit the concurrent processes in order to prevent it.


Using the Cube Builder
----------------------

Please, refer to the document `USING.rst <./USING.rst>`_ for information on how to use the cube builder.



.. rubric:: Footnotes

.. [#f1]

    During GDAL installation, if you have a build message such as the one showed below:

    .. code-block::

        Skipping optional fixer: ws_comma
        running build_ext
        building 'osgeo._gdal' extension
        creating build/temp.linux-x86_64-3.7
        creating build/temp.linux-x86_64-3.7/extensions
        x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I../../port -I../../gcore -I../../alg -I../../ogr/ -I../../ogr/ogrsf_frmts -I../../gnm -I../../apps -I/home/gribeiro/Devel/github/brazil-data-cube/cube-builder/venv/include -I/usr/include/python3.7m -I. -I/usr/include -c extensions/gdal_wrap.cpp -o build/temp.linux-x86_64-3.7/extensions/gdal_wrap.o
        extensions/gdal_wrap.cpp:3168:10: fatal error: cpl_port.h: No such file or directory
         #include "cpl_port.h"
                  ^~~~~~~~~~~~
        compilation terminated.
        error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
        Running setup.py install for gdal ... error
        Cleaning up...

    You can instruct ``pip`` to look at the right place for header files when building GDAL:

    .. code-block:: shell

        $ C_INCLUDE_PATH="/usr/include/gdal" \
          CPLUS_INCLUDE_PATH="/usr/include/gdal" \
          pip install "gdal==2.4.2"


.. [#f2]

    On Linux Ubuntu 18.04 LTS you can install GDAL 2.4.2 from the UbuntuGIS repository:

    1. Create a file named ``/etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-bionic.list`` and add the following content:

    .. code-block:: shell

        deb http://ppa.launchpad.net/ubuntugis/ppa/ubuntu bionic main
        deb-src http://ppa.launchpad.net/ubuntugis/ppa/ubuntu bionic main


    2. Then add the following key:

    .. code-block:: shell

        $ sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6B827C12C2D425E227EDCA75089EBE08314DF160


    3. Then, update your repository index:

    .. code-block:: shell

        $ sudo apt-get update


    4. Finally, install GDAL:

    .. code-block:: shell

        $ sudo apt-get install libgdal-dev=2.4.2+dfsg-1~bionic0


.. [#f3]

    During ``librabbitmq`` installation, if you have a build message such as the one showed below:

    .. code-block::

        ...
        Running setup.py install for SQLAlchemy-Utils ... done
        Running setup.py install for bdc-db ... done
        Running setup.py install for librabbitmq ... error
        ERROR: Command errored out with exit status 1:
         command: /home/gribeiro/Devel/github/brazil-data-cube/bdc-collection-builder/venv/bin/python3.7 -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-1i7mp5js/librabbitmq/setup.py'"'"'; __file__='"'"'/tmp/pip-install-1i7mp5js/librabbitmq/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record /tmp/pip-record-m9lm5kjn/install-record.txt --single-version-externally-managed --compile --install-headers /home/gribeiro/Devel/github/brazil-data-cube/bdc-collection-builder/venv/include/site/python3.7/librabbitmq
             cwd: /tmp/pip-install-1i7mp5js/librabbitmq/
        Complete output (107 lines):
        /tmp/pip-install-1i7mp5js/librabbitmq/setup.py:167: DeprecationWarning: 'U' mode is deprecated
          long_description = open(os.path.join(BASE_PATH, 'README.rst'), 'U').read()
        running build
        - pull submodule rabbitmq-c...
        Cloning into 'rabbitmq-c'...
        Note: checking out 'caad0ef1533783729c7644a226c989c79b4c497b'.

        You are in 'detached HEAD' state. You can look around, make experimental
        changes and commit them, and you can discard any commits you make in this
        state without impacting any branches by performing another checkout.

        If you want to create a new branch to retain commits you create, you may
        do so (now or later) by using -b with the checkout command again. Example:

          git checkout -b <new-branch-name>

        - autoreconf
        sh: 1: autoreconf: not found
        - configure rabbitmq-c...
        /bin/sh: 0: Can't open configure


    You will need to install ``autoconf``:

    .. code-block:: shell

        $ sudo apt install autoconf