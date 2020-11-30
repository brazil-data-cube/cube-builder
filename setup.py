#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
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

extras_require = {
    'docs': docs_require,
    'tests': tests_require,
}

extras_require['all'] = [ req for exts, reqs in extras_require.items() for req in reqs ]

setup_requires = []

install_requires = [
    'bdc-catalog @ git+git://github.com/brazil-data-cube/bdc-catalog.git@v0.6.4#egg=bdc-catalog',
    'celery[librabbitmq]>=4.3.0,<5',
    'Flask>=1.1.1,<2',
    'marshmallow-sqlalchemy>=0.19.0,<1',
    'numpy>=1.17.2',
    'numpngw>=0.0.8',
    'python-dateutil>=2.8,<3',
    'rasterio[s3]>=1.1.2,<2',
    'rio_cogeo>=1.1,<2',
    'shapely>=1.7,<2',
    'SQLAlchemy-Utils>=0.34.2,<1',
    'stac.py==0.9.0.post5',
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
    keywords=('Cube Builder', 'Datacube', ),
    license='MIT',
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
        'console_scripts': [
            'cube-builder = cube_builder.cli:cli'
        ]
    },
    extras_require=extras_require,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Web Environment',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
