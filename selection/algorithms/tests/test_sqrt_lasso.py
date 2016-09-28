from __future__ import division

import numpy as np
import numpy.testing.decorators as dec
import nose.tools as nt

import regreg.api as rr

from selection.algorithms.sqrt_lasso import (solve_sqrt_lasso, 
                                             choose_lambda,
                                             goodness_of_fit,
                                             sqlasso_objective,
                                             sqlasso_objective_skinny,
                                             solve_sqrt_lasso_fat,
                                             solve_sqrt_lasso_skinny)

from selection.tests.instance import gaussian_instance as instance
from selection.algorithms.lasso import lasso
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_for_test

@set_sampling_params_iftrue(True)
@dec.slow
def test_goodness_of_fit(n=20, p=25, s=10, sigma=20.,
                         nsim=1000, burnin=2000, ndraw=8000):
    P = []
    while True:
        y = np.random.standard_normal(n) * sigma
        beta = np.zeros(p)
        X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
        X /= (X.std(0)[None,:] * np.sqrt(n))
        y += np.dot(X, beta) * sigma
        lam_theor = .7 * choose_lambda(X, quantile=0.9)
        L = lasso.sqrt_lasso(X, y, lam_theor)
        L.fit()
        pval = goodness_of_fit(L, 
                               lambda x: np.max(np.fabs(x)),
                               burnin=burnin,
                               ndraw=ndraw)
        P.append(pval)
        Pa = np.array(P)
        Pa = Pa[~np.isnan(Pa)]
        if (~np.isnan(np.array(Pa))).sum() >= nsim:
            break

    # make any plots not use display

    from matplotlib import use
    use('Agg')
    import matplotlib.pyplot as plt

    # used for ECDF

    import statsmodels.api as sm

    U = np.linspace(0,1,101)
    plt.plot(U, sm.distributions.ECDF(Pa)(U))
    plt.plot([0,1], [0,1])
    plt.savefig("goodness_of_fit_uniform", format="pdf")
    
@set_seed_for_test(10)
def test_skinny_fat():

    X, Y = instance()[:2]
    n, p = X.shape
    lam = choose_lambda(X)
    obj1 = sqlasso_objective(X, Y)
    obj2 = sqlasso_objective_skinny(X, Y)
    soln1 = solve_sqrt_lasso_fat(X, Y, weights=np.ones(p) * lam, solve_args={'min_its':500})[0]
    soln2 = solve_sqrt_lasso_skinny(X, Y, weights=np.ones(p) * lam, solve_args={'min_its':500})[0]

    np.testing.assert_allclose(soln1, soln2, rtol=1.e-3)

    X, Y = instance(p=50)[:2]
    n, p = X.shape
    lam = choose_lambda(X)
    obj1 = sqlasso_objective(X, Y)
    obj2 = sqlasso_objective_skinny(X, Y)
    soln1 = solve_sqrt_lasso_fat(X, Y, weights=np.ones(p) * lam, solve_args={'min_its':500})[0]
    soln2 = solve_sqrt_lasso_skinny(X, Y, weights=np.ones(p) * lam, solve_args={'min_its':500})[0]

    np.testing.assert_allclose(soln1, soln2, rtol=1.e-3)

