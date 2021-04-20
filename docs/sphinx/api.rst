..
    This file is part of Python Module for Cube Builder.
    Copyright (C) 2019-2021 INPE.

    Cube Builder free software; you can redistribute it and/or modify it
    under the terms of the MIT License; see LICENSE file for more details.


API Reference
=============


.. module:: cube_builder


Data Cube Warp
--------------

.. autofunction:: cube_builder.utils.processing.merge


Data Cube Composition
---------------------

.. autofunction:: cube_builder.utils.processing.blend


Utils for Image Operation
-------------------------

.. autofunction:: cube_builder.utils.processing.compute_data_set_stats

.. automodule:: cube_builder.utils.image
    :members:

Band Index Generator
--------------------

.. automodule:: cube_builder.utils.index_generator
    :members:

Tasks
-----

.. automodule:: cube_builder.celery.worker
    :members:

.. automodule:: cube_builder.celery.tasks
    :members:

    The processing workflow consists in::

        Search    ->    Merge    ->    prepare_blend    ->    blend    ->    publish


    .. autofunction:: warp_merge
    .. autofunction:: prepare_blend
    .. autofunction:: blend
    .. autofunction:: publish