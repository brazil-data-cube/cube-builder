..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2020 INPE.

    Cube Builder is free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


Installation
============

``cube-buider`` has essentially several dependencies. Please, read the instructions below in order to install ``cube-builder``.


Production installation
-----------------------

**Under Development!**

.. Install from `PyPI <https://pypi.org/>`_:
..
.. .. code-block:: shell
..
..     $ pip3 install cube-builder


Development installation
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


Generate the documentation:

.. code-block:: shell

        $ python setup.py build_sphinx


The above command will generate the documentation in HTML and it will place it under:

.. code-block:: shell

    build/sphinx/html/
