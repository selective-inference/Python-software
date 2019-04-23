"""
simple example script for running notebooks and saving the resulting notebook.

Usage: `strip_notebook.py` foo.ipynb [bar.ipynb [...]]`

Each notebook is stripped of its outputs after checking that it executes.
Used to clean notebooks before committing to git.
"""

from selection.utils.nbtools import strip_outputs, reads, writes
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(
        description='Run cells in notebook and strip outputs.')
    parser.add_argument('--clobber', action='store_true',
                        help='if set, overwrite existing notebook files with stripped version')
    parser.add_argument('--norun', action='store_true',
                        help='if set, do not run cells before stripping')
    parser.add_argument('notebooks',
                        metavar='NB',
                        help='Notebooks to strip outputs from.',
                        nargs='+',
                        type=str)

    args = parser.parse_args()

    for ipynb in args.notebooks:
        print("running and stripping %s" % ipynb)
        with open(ipynb) as f:
            stripped_nb = strip_outputs(reads(f.read(), 'json'),
                                        run_cells=not args.norun)
        if args.clobber:
            print('clobbering %s' % ipynb)
            with open(ipynb, 'w') as f:
                f.write(writes(stripped_nb, 'json'))
        else:
            print('not clobbering %s' % ipynb)

if __name__ == '__main__':
    main()
