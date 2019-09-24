#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Script to auto-generate our API docs.

This script should run in Python 2 and Python 3
"""
# stdlib imports
import os

# local imports
from apigen import ApiDocWriter

#*****************************************************************************
if __name__ == '__main__':
    package = 'selectinf'
    outdir = os.path.join('source', 'api', 'generated')
    docwriter = ApiDocWriter(package)
    docwriter.package_skip_patterns += [r'\.fixes$',
                                        r'\.externals$',
                                        #r'\.labs\.viz',
                                        ]
    docwriter.write_api_docs(outdir)
    docwriter.write_index(outdir, 'gen', relative_to=os.path.join('source', 'api'))
    print('%d files written' % len(docwriter.written_modules))
