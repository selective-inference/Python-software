#!/usr/bin/env python
''' Installation script for selection package '''

import os
import sys
from os.path import join as pjoin, dirname
from setup_helpers import package_check

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import numpy as np

# Get version and release info, which is all stored in regreg/info.py
ver_file = os.path.join('selection', 'info.py')
# Use exec for compabibility with Python 3
exec(open(ver_file).read())

from distutils.command import install
from distutils.core import setup
from distutils.extension import Extension

from cythexts import cyproc_exts, get_pyx_sdist
from setup_helpers import package_check

# Define extensions
EXTS = []
for modulename, other_sources in (
    ('selection.sampling.truncnorm', []),
    ('selection.sampling.sqrt_lasso', []),
    ):
    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename,[pyx_src] + other_sources,
                          include_dirs = [np.get_include(),
                                         "src"],
                          libraries=['m']),
                )
extbuilder = cyproc_exts(EXTS, CYTHON_MIN_VERSION, 'pyx-stamps')

extra_setuptools_args = {}

class installer(install.install):
    def run(self):
        package_check('numpy', NUMPY_MIN_VERSION)
        package_check('scipy', SCIPY_MIN_VERSION)
        package_check('sklearn', SKLEARN_MIN_VERSION)
        package_check('mpmath', MPMATH_MIN_VERSION)
        install.install.run(self)

cmdclass = dict(
    build_ext=extbuilder,
    install=installer,
    sdist=get_pyx_sdist()
)


def main(**extra_args):
    setup(name=NAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          url=URL,
          download_url=DOWNLOAD_URL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          platforms=PLATFORMS,
          version=VERSION,
          requires=REQUIRES,
          provides=PROVIDES,
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
                          'selection.tests'
                          ],
          ext_modules = EXTS,
          package_data = {},
          data_files=[],
          scripts= [],
          cmdclass = cmdclass,
          **extra_args
         )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main(**extra_setuptools_args)
