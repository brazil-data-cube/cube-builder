#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019 INPE.
#
# Cube Builder free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Python Module for Cube Builder."""

import os
from setuptools import find_packages, setup

readme = open('README.rst').read()

history = open('CHANGES.rst').read()

docs_require = [
    'Sphinx>=2.2',
]

tests_require = []

extras_require = {
    'docs': docs_require,
    'tests': tests_require,
}

extras_require['all'] = [ req for exts, reqs in extras_require.items() for req in reqs ]

setup_requires = []

install_requires = [
    'bdc-core @ git+git://github.com/brazil-data-cube/bdc-core.git#egg=bdc-core',
    'bdc-db @ git+git://github.com/brazil-data-cube/bdc-db.git#egg=bdc-db',
    'celery[librabbitmq]>=4.3.0',
    'Flask>=1.1.1',
    'GDAL>=2.3.3',
    'marshmallow-sqlalchemy>=0.19.0',
    'numpy>=1.17.2',
    'numpngw>=0.0.8',
    'rasterio>=1.1.2',
    'SQLAlchemy-Utils>=0.34.2',
    'stac @ git+git://github.com/brazil-data-cube/stac.py.git@v0.7#egg=stac',
    'scikit-image>=0.16.2',
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
    author='INPE',
    author_email='admin@admin.com',
    url='https://github.com/brazil-data-cube/cube-builder',
    packages=packages,
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    entry_points={
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
