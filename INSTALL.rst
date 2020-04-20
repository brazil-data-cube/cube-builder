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


Using pip
~~~~~~~~~

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
        ``15672`` the RabbitMQ User Interface. You can check `<http://127.0.0.1:15672>`_. The default credentiais are ``guest`` and ``guest`` for
        user and password respectively.



Prepare and initialize database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    The ``cube-builder`` uses `bdc-db <https://github.com/brazil-data-cube/bdc-db/>`_ as database definition to store data cube metadata.
    Make sure you have a prepared database on PostgreSQL. You can follow steps `here <https://github.com/brazil-data-cube/bdc-db/blob/master/RUNNING.rst>`_.


Edit file **cube_builder/config.py** the following variables:

1. **SQLALCHEMY_DATABASE_URI** URI Connection to database
2. **DATA_DIR** Path to store datacubes. Make sure the directory exists.

.. code-block:: shell

        cube-builder db create # Create database and schema
        cube-builder db upgrade # Up migrations



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

Enter the following command to start cube-builder worker

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