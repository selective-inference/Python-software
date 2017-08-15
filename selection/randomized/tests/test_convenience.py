from itertools import product
import numpy as np
import nose.tools as nt

from ..convenience import lasso, step, threshold
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)

def test_lasso_constructors():

    cls = lasso
    for const_info, rand in product(zip([gaussian_instance,
                                         logistic_instance,
                                         poisson_instance],
                                        [cls.gaussian,
                                         cls.logistic,
                                         cls.poisson]),
                              ['gaussian', 'logistic', 'laplace']):

        inst, const = const_info
        X, Y = inst()[:2]
        W = np.ones(X.shape[1])
        conv = const(X, Y, W, randomizer=rand)
        conv.fit()

def test_step_constructors():

    cls = step
    for const_info, rand in product(zip([gaussian_instance,
                                         logistic_instance,
                                         poisson_instance],
                                        [cls.gaussian,
                                         cls.logistic,
                                         cls.poisson]),
                              ['gaussian', 'logistic', 'laplace']):

        inst, const = const_info
        X, Y = inst()[:2]
        W = np.ones(X.shape[1])
        conv = const(X, Y, W)
        conv.fit()

        n, p = X.shape
        active = np.zeros(p, np.bool)
        active[:int(p/2)] = True

        conv = const(X, Y, W, active=active)
        conv.fit()

        conv = const(X, Y, W, inactive=~active)
        conv.fit()
        
        conv = const(X, Y, W, inactive=~active, active=active)
        conv.fit()
        

def test_threshold_constructors():

    cls = threshold
    for const_info, rand in product(zip([gaussian_instance,
                                         logistic_instance,
                                         poisson_instance],
                                        [cls.gaussian,
                                         cls.logistic,
                                         cls.poisson]),
                              ['gaussian', 'logistic', 'laplace']):

        inst, const = const_info
        X, Y = inst()[:2]
        W = np.ones(X.shape[1])

        n, p = X.shape
        active = np.zeros(p, np.bool)
        active[:int(p/2)] = True

        conv = const(X, Y, W, active=active)
        conv.fit()

        conv = const(X, Y, W, inactive=~active)
        conv.fit()
        
        conv = const(X, Y, W, inactive=~active, active=active)
        conv.fit()
        
