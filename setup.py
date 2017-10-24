#!/usr/bin/env python
''' Installation script for selection package '''

import os
import sys
from os.path import join as pjoin, dirname, exists

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

# Unconditionally require setuptools
import setuptools

# Package for getting versions from git tags
import versioneer

# Import distutils _after_ setuptools import, and after removing
# MANIFEST
from distutils.core import setup
from distutils.extension import Extension

from cythexts import cyproc_exts, get_pyx_sdist
from setup_helpers import (SetupDependency, read_vars_from,
                           make_np_ext_builder)

# Get various parameters for this version, stored in selection/info.py
info = read_vars_from(pjoin('selection', 'info.py'))

# Try to preempt setuptools monkeypatching of Extension handling when Pyrex
# is missing.  Otherwise the monkeypatched Extension will change .pyx
# filenames to .c filenames, and we probably don't have the .c files.
sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))
# Set setuptools extra arguments
extra_setuptools_args = dict(
    tests_require=['nose'],
    test_suite='nose.collector',
    zip_safe=False,
    extras_require = dict(
        doc=['Sphinx>=1.0'],
        test=['nose>=0.10.1']))

# Define extensions
EXTS = []
for modulename, other_sources in (
    ('selection.sampling.truncnorm', []),
    ('selection.sampling.truncnorm_quadratic', []),
    ('selection.sampling.sqrt_lasso', []),
    ):
    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename,[pyx_src] + other_sources,
                          libraries=['m']),
                )
