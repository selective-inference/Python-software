.. _download:

Downloading and installing the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The post-selection inference source code is hosted at 

http://github.com/selective-inference/Python-software

Selection depends on the following Python tools

* `NumPy <http://numpy.scipy.org>`_

* `SciPy <http://www.scipy.org>`_

* `Cython <http://www.cython.org>`_

* `Pandas <http://www.pandas.org>`_

You can clone the selection repo using::

     git clone https://github.com/selective-inference/Python-software.git

Then installation is a simple call to python::

     cd selection
     git submodule update --init
     pip install -r requirements.txt
     python setup.py install --prefix=MYDIR

where MYDIR is a site-packages directory you can write to. This
directory will need to be on your PYTHONPATH for you to import
`selectinf`. That's it!

Testing your installation
-------------------------

There is a small but growing suite of tests that be easily checked using `nose <http://somethingaboutorange.com/mrl/projects/nose/1.0.0/>`_::

     mkdir tmp
     cd tmp
     nosetests -v selectinf

