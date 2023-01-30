#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#

"""Python Module for Cube Builder."""

import os

from setuptools import find_packages, setup

readme = open('README.rst').read()

history = open('CHANGES.rst').read()

docs_require = [
    'Sphinx>=2.1.2',
    'sphinx_rtd_theme',
    'sphinx-copybutton',
]

tests_require = [
    'coverage>=4.5',
    'coveralls>=1.8',
    'pytest>=5.2',
    'pytest-cov>=2.8',
    'pytest-pep8>=1.0',
    'pydocstyle>=4.0',
    'isort>4.3',
    'check-manifest>=0.40',
]

histogram_require = [
    'scikit-image>=0.18,<1'
]

extras_require = {
    'docs': docs_require,
    'tests': tests_require,
    'histogram': histogram_require,
    'rabbitmq': [
        'librabbitmq>=1.5.0',
    ]
}

extras_require['all'] = [ req for exts, reqs in extras_require.items() for req in reqs ]

setup_requires = []

install_requires = [
    'bdc-catalog @ git+https://github.com/brazil-data-cube/bdc-catalog.git@v0.8.2#egg=bdc-catalog',
    'celery>=4.3.0,<5',
    'Flask>=1.1.1,<2',
    'flask-redoc>=0.2.1',
    'marshmallow-sqlalchemy>=0.19.0,<0.25',
    'numpy>=1.17.2',
    'numpngw>=0.0.8',
    'python-dateutil>=2.8,<3',
    'pyproj==3.3.1',
    'rasterio[s3]==1.2.1',
    'requests>=2.25.1',
    'rio_cogeo==3.0.2',
    'shapely>=1.7,<2',
    'SQLAlchemy-Utils>=0.34.2,<1',
    'pystac-client>=0.5',
    'MarkupSafe==2.0.1',
    'bdc-auth-client @ git+https://github.com/brazil-data-cube/bdc-auth-client.git@v0.2.1#egg=bdc-auth-client'
]

packages = find_packages()

with open(os.path.join('cube_builder', 'version.py'), 'rt') as fp:
    g = {}
    exec(fp.read(), g)
    version = g['__version__']

setup(
    name='cube-builder',
    version=version,
    description=__doc__,
    long_description=readme + '\n\n' + history,
    keywords=['Cube Builder', 'Datacube'],
    license='GPLv3',
    author='Brazil Data Cube Team',
    author_email='brazildatacube@inpe.br',
    url='https://github.com/brazil-data-cube/cube-builder',
    packages=packages,
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    entry_points={
        'bdc_db.alembic': [
            'cube_builder = cube_builder:alembic'
        ],
        'bdc_db.models': [
            'cube_builder = cube_builder.models'
        ],
        'bdc_db.namespaces': [
            'cube_builder = cube_builder.config:Config.ACTIVITIES_SCHEMA'
        ],
        'console_scripts': [
            'cube-builder = cube_builder.cli:cli'
        ]
    },
    extras_require=extras_require,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    classifiers=[
        'Development Status :: 1 - Stable',
        'Environment :: Web Environment',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL v3 License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
