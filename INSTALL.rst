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


Installation
============

The ``Cube Builder`` depends essentially on:

- `Python Client Library for STAC (stac.py) <https://github.com/brazil-data-cube/stac.py>`_

- `Flask <https://palletsprojects.com/p/flask/>`_

- `Celery <http://www.celeryproject.org/>`_

- `rasterio <https://rasterio.readthedocs.io/en/latest/>`_

- `NumPy <https://numpy.org/>`_

- `scikit-image <https://scikit-image.org/>`_

- `RabbitMQ <https://www.rabbitmq.com/>`_

- `marshmallow-SQLAlchemy <https://marshmallow-sqlalchemy.readthedocs.io/en/latest/>`_

- `Brazil Data Cube Catalog Module <https://github.com/brazil-data-cube/bdc-catalog.git>`_


Compatibility
-------------

+--------------+-------------+
| Cube-Builder | BDC-Catalog |
+==============+=============+
| 0.8.x        | 0.8.2       |
+--------------+-------------+
| 0.4.x, 0.6.x | 0.8.1       |
+--------------+-------------+
| 0.2.x        | 0.2.x       |
+--------------+-------------+

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

    $ pip3 install -U pip setuptools wheel
    $ pip3 install -e .[all]


.. note::

    If you have problems with the ``librabbitmq`` installation, please, see [#f1]_.


Running in Development Mode
---------------------------


Launch the RabbitMQ Container
*****************************


You will need an instance of RabbitMQ up and running in order to launch the ``cube-builder`` celery workers.


We have prepared in the ``Cube Builder`` repository a configuration for ``RabbitMQ`` container with ``docker-compose``. Please, follow the steps below::

    docker-compose up -d mq


After that command, check which port was binded from the host to the container::

    $ docker container ls

    CONTAINER ID   IMAGE                  COMMAND                  CREATED         STATUS         PORTS                    NAMES
    a3bb86d2df56   rabbitmq:3-management  "docker-entrypoint.s…"   3 minutes ago   Up 3 minutes   4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp   cube-builder-rabbitmq


.. note::

    In the above output the ``RabbitMQ`` service is attached to the ports ``5672`` for socket client and ``15672`` for the RabbitMQ User Interface. You can check `<http://127.0.0.1:15672>`_. The default credentials are ``guest`` and ``guest`` for ``user`` and ``password`` respectively.


Prepare the Database System
***************************


The ``Cube Builder`` uses `BDC-DB <https://github.com/brazil-data-cube/bdc-db/>`_ as database definition to store data cube metadata.


.. note::

    If you already have a database instance with the Brazil Data Cube data model, you can skip this section.

In order to proceed with installation, you will need ``PostgreSQL with PostGIS``. We have already prepared a minimal
instance in ``docker-compose.yml``. You may use it as following::

    docker-compose up -d postgres


We have prepared a script to configure the database model::

    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" ./deploy/configure-db.sh



Launch the ``Cube Builder`` service
***********************************


In the source code folder, enter the following command::

    FLASK_ENV="development" \
    WORK_DIR="/workdir" \
    DATA_DIR="/data" \
    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder run


You may need to replace the definition of some environment variables:

- ``FLASK_ENV="development"``: used to tell Flask to run in ``Debug`` mode.

- ``WORK_DIR="/workdir"``: set path to store temporary cubes/processing.

- ``DATA_DIR="/data"``: set path to store data cubes

- ``SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc"``: set the database URI connection for PostgreSQL.


The above command should output some messages in the console as showed below::

    * Environment: development
    * Debug mode: on
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    * Restarting with stat
    * Debugger is active!
    * Debugger PIN: 319-592-254


Launch the ``Cube Builder`` worker
**********************************


Enter the following command to start ``Cube Builder`` worker::

    WORK_DIR="/workdir" \
    DATA_DIR="/data" \
    SQLALCHEMY_DATABASE_URI="postgresql://postgres:postgres@localhost/bdc" \
    cube-builder worker -l INFO --concurrency 8 -Q default,merge-cube,prepare-cube,blend-cube,publish-cube


You may need to replace the definition of some parameters:

    - ``-l INFO``: defines the ``Logging level``. You may choose between ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``, or ``FATAL``.

    - ``--concurrency 8``: defines the number of concurrent processes to generate of data cube. The default is the number of CPUs available on your system.

    - ``-Q default,merge-cube,prepare-cube,blend-cube,publish-cube``: the list of Queues to be consumed by ``Cube-Builder`` in order to execute the tasks generation. You can set many workers to listen specific queues and set the maximum of threads to be executed in parallel.

.. note::

    The command line ``cube-builder worker`` is an auxiliary tool that wraps celery command line
    using ``cube_builder`` as context. In this way, all ``celery worker`` parameters are currently supported.
    See more in `Celery Workers Guide <https://docs.celeryproject.org/en/stable/userguide/workers.html>`_.


.. warning::

    The ``Cube Builder`` can use a lot of memory for each concurrent process, since it opens multiple images in memory.
    You can limit the concurrent processes in order to prevent it.


.. rubric:: Footnotes


.. [#f1]

    During ``librabbitmq`` installation, if you have a build message such as the one showed below:

    .. code-block::

        ...
        Running setup.py install for SQLAlchemy-Utils ... done
        Running setup.py install for bdc-db ... done
        Running setup.py install for librabbitmq ... error
        ERROR: Command errored out with exit status 1:
         command: /home/gribeiro/Devel/github/brazil-data-cube/bdc-collection-builder/venv/..
             cwd: /tmp/pip-install-1i7mp5js/librabbitmq/
        Complete output (107 lines):
        /tmp/pip-install-1i7mp5js/librabbitmq/setup.py:167:
        DeprecationWarning: 'U' mode is deprecated
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