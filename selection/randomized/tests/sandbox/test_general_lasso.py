from itertools import product
import numpy as np
import nose.tools as nt

from ..lasso import lasso
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from ...tests.flags import SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue 

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=50, burnin=20)
def test_lasso_constructors(ndraw=1000, burnin=200):
    """
    Smoke tests for lasso convenience constructors
    """
    cls = lasso
    for const_info, rand, marginalize, condition in product(zip([gaussian_instance,
                                                                 logistic_instance,
                                                                 poisson_instance],
                                                                [cls.gaussian,
                                                                 cls.logistic,
                                                                 cls.poisson]),
                                                            ['gaussian', 'logistic', 'laplace'],
                                                            [False, True],
                                                            [False, True]):

        print(rand)
        inst, const = const_info
        X, Y = inst(n=100, p=20, signal=5, s=10)[:2]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 0.2
        W[0] = 0
        W[3:] = 50.
        np.random.shuffle(W)
        conv = const(X, Y, W, randomizer=rand)
        nboot = 1000
        if SMALL_SAMPLES:
            nboot = 20
        signs = conv.fit(nboot=nboot)

        marginalize = None
        if marginalize:
            marginalize = np.zeros(p, np.bool)
            marginalize[:int(p/2)] = True
        
        condition = None
        if condition:
            if marginalize:
                condition = ~marginalize
            else:
                condition = np.ones(p, np.bool)
            condition[-int(p/4):] = False

        selected_features = np.zeros(p, np.bool)
        selected_features[:3] = True

        conv.summary(selected_features,
                     ndraw=ndraw,
                     burnin=burnin,
                     compute_intervals=True)

        conv.decompose_subgradient(marginalize=marginalize,
                                   condition=condition)

        conv.summary(selected_features,
                     ndraw=ndraw,
                     burnin=burnin)

        conv.decompose_subgradient(condition=np.ones(p, np.bool))

        conv.summary(selected_features,
                     ndraw=ndraw,
                     burnin=burnin)
