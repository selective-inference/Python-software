.. _download:

Downloading and installing the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The post-selection inference source code is hosted at 

http://bitbucket.org/jetaylor74/selection

Selection depends on the following Python tools

* `NumPy <http://numpy.scipy.org>`_

* `SciPy <http://www.scipy.org>`_

* `Cython <http://www.cython.org>`_

You can clone the selection repo using::

     git clone https://bitbucket.org/jetaylor74/selection.git

Then installation is a simple call to python::

     cd selection
     python setup.py install --prefix=MYDIR

where MYDIR is a site-packages directory you can write to. This directory will need to be on your PYTHONPATH for you to import `selection`. That's it!

Testing your installation
-------------------------

There is a small but growing suite of tests that be easily checked using `nose <http://somethingaboutorange.com/mrl/projects/nose/1.0.0/>`_::

     cd selection/tests
     nosetests

