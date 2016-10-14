from __future__ import division

import numpy as np
import numpy.testing.decorators as dec
import nose.tools as nt

import regreg.api as rr

from selection.tests.instance import gaussian_instance as instance
from selection.tests.decorators import (set_sampling_params_iftrue, 
                                        set_seed_iftrue, 
                                        wait_for_return_value,
                                        register_report)
import selection.tests.reports as reports

from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.algorithms.sqrt_lasso import (solve_sqrt_lasso, 
                                             choose_lambda,
                                             goodness_of_fit,
                                             sqlasso_objective,
                                             sqlasso_objective_skinny,
                                             solve_sqrt_lasso_fat,
                                             solve_sqrt_lasso_skinny)


from selection.algorithms.lasso import lasso

@register_report(['pvalue', 'active'])
@wait_for_return_value()
@set_sampling_params_iftrue(SMALL_SAMPLES, nsim=10, burnin=10, ndraw=10)
@dec.slow
def test_goodness_of_fit(n=20, p=25, s=10, sigma=20.,
                         nsim=10, burnin=2000, ndraw=8000):
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

    return Pa, np.zeros_like(Pa, np.bool)
    
@set_seed_iftrue(SET_SEED)
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

def report(niter=50, **kwargs):

    _report = goodness_of_fit_report = reports.reports['test_goodness_of_fit']
    runs = reports.collect_multiple_runs(_report['test'],
                                         _report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)
    fig = reports.pvalue_plot(runs)
    fig.savefig('sqrtlasso_goodness_of_fit.pdf')


