#!/usr/bin/env python
''' Installation script for fixed_lambda package '''

import os
import sys
from os.path import join as pjoin, dirname
from setup_helpers import package_check

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import numpy as np

# Get version and release info, which is all stored in regreg/info.py
ver_file = os.path.join('fixed_lambda', 'info.py')
# Use exec for compabibility with Python 3
exec(open(ver_file).read())

from distutils.command import install
from distutils.core import setup
extra_setuptools_args = {}
EXTS = []

class installer(install.install):
    def run(self):
        package_check('numpy', NUMPY_MIN_VERSION)
        package_check('scipy', SCIPY_MIN_VERSION)
        install.install.run(self)


cmdclass = dict(
    install=installer)


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
          packages     = ['fixed_lambda',
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
