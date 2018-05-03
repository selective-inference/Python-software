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

EXTS.append(Extension('selection.algorithms.debiased_lasso_utils',
                      ['selection/algorithms/debiased_lasso_utils.pyx',
                       'C-software/src/quadratic_program_wide.c'],
                      libraries=['m'],
                      include_dirs=['C-software/src']))

EXTS.append(Extension('selection.randomized.selective_MLE_utils',
                      ['selection/randomized/selective_MLE_utils.pyx',
                       'C-software/src/selective_mle.c'],
                      libraries=['m'],
                      include_dirs=['C-software/src']))

# Cython is a dependency for building extensions, iff we don't have stamped
# up pyx and c files.
build_ext, need_cython = cyproc_exts(EXTS,
                                     info.CYTHON_MIN_VERSION,
                                     'pyx-stamps')

# Add numpy includes when building extension.
build_ext = make_np_ext_builder(build_ext)

# Check dependencies, maybe add to setuptools lists
if need_cython:
    SetupDependency('Cython', info.CYTHON_MIN_VERSION,
                    req_type='install_requires',
                    heavy=False).check_fill(extra_setuptools_args)
SetupDependency('numpy', info.NUMPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('scipy', info.SCIPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)


cmdclass=versioneer.get_cmdclass()
cmdclass.update(dict(
    build_ext=build_ext,
    sdist=get_pyx_sdist()))


def main(**extra_args):
    setup(name=info.NAME,
          maintainer=info.MAINTAINER,
          maintainer_email=info.MAINTAINER_EMAIL,
          description=info.DESCRIPTION,
          url=info.URL,
          download_url=info.DOWNLOAD_URL,
          license=info.LICENSE,
          classifiers=info.CLASSIFIERS,
          author=info.AUTHOR,
          author_email=info.AUTHOR_EMAIL,
          platforms=info.PLATFORMS,
          version=versioneer.get_version(),
          requires=info.REQUIRES,
          provides=info.PROVIDES,
          packages     = ['selection',
                          'selection.utils',
                          'selection.truncated',
                          'selection.truncated.tests',
                          'selection.constraints',
                          'selection.constraints.tests',
                          'selection.distributions',
                          'selection.distributions.tests',
                          'selection.algorithms',
                          'selection.algorithms.tests',
                          'selection.sampling',
                          'selection.sampling.tests',
                          'selection.randomized',
                          'selection.randomized.tests',
                          'selection.tests'
                          ],
          ext_modules = EXTS,
          package_data = {},
          data_files=[],
          scripts= [],
          long_description = open('README.md', 'rt').read(),
          cmdclass = cmdclass,
          **extra_args
         )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main(**extra_setuptools_args)
