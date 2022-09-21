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