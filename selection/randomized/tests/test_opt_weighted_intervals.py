from itertools import product
import numpy as np
import nose.tools as nt

from ..convenience import lasso, step, threshold
from ..query import optimization_sampler
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from ...tests.flags import SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
import matplotlib.pyplot as plt

from scipy.stats import t as tdist
from ..glm import glm_nonparametric_bootstrap, pairs_bootstrap_glm
from ..M_estimator import restricted_Mest

@set_seed_iftrue(False, 200)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=1000, burnin=100)
def test_opt_weighted_intervals(ndraw=20000, burnin=2000):

    results = []
    cls = lasso
    for const_info, rand in product(zip([gaussian_instance], [cls.gaussian]), ['laplace', 'gaussian']):

        inst, const = const_info

        X, Y, beta = inst(n=100, p=20, s=0, signal=5., sigma=5.)[:3]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 8
        conv = const(X, Y, W, randomizer=rand, parametric_cov_estimator=True)
        signs = conv.fit()
        print("signs", signs)

        marginalizing_groups = np.ones(p, np.bool)
        #marginalizing_groups[:int(p/2)] = True
        conditioning_groups = ~marginalizing_groups
        #conditioning_groups[-int(p/4):] = False
        conv.decompose_subgradient(marginalizing_groups=marginalizing_groups,
                                   conditioning_groups=conditioning_groups)

        selected_features = conv._view.selection_variable['variables']
        nactive=selected_features.sum()
        print("nactive", nactive)
        if nactive==0:
            results.append(None)
        else:
            sel_pivots, sel_pval, sel_ci = conv.summary(selected_features,
                                                        parameter=beta[selected_features],
                                                        ndraw=ndraw,
                                                        burnin=burnin,
                                                        compute_intervals=True)
            print(sel_pivots)
            results.append((rand, sel_pivots, sel_ci, beta[selected_features]))

    return results



