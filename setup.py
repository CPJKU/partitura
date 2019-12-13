#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'partitura'
DESCRIPTION = 'A package for handling symbolic musical information'
KEYWORDS = 'music notation musicxml midi'
URL = 'https://github.com/mgrachten/partitura'
EMAIL = 'partitura-users@googlegroups.com'
AUTHOR = 'Maarten Grachten, Carlos Cancino-Chacón, and Thassilo Gadermaier'
REQUIRES_PYTHON = '>=3.5'
VERSION = '0.3.5'

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy',
    'scipy',
    'lxml',
    'lark-parser',
    'xmlschema',
    'mido'
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    # long_description_content_type='text/markdown',
    long_description_content_type='text/x-rst',
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'partitura': ['assets/musicxml.xsd',
                                'assets/score_example.musicxml']},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Apache 2.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
