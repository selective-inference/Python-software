from __future__ import print_function
import sys

import numpy as np
import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.approx_ci.ci_approx_density import approximate_conditional_density
from selection.approx_ci.selection_map import M_estimator_map

def test_approximate_MLE(X,
                         y,
                         true_mean,
                         sigma,
                         seed_n = 0,
                         lam_frac = 1.,
                         loss='gaussian',
                         randomization_scale = 1.):
    from selection.api import randomization

    n, p = X.shape
    np.random.seed(seed_n)
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=randomization_scale)
    M_est = M_estimator_map(loss, epsilon, penalty, randomization, randomization_scale=randomization_scale)

    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
    sys.stderr.write("Observed target" + str(M_est.target_observed) + "\n")

    ci = approximate_conditional_density(M_est)
    ci.solve_approx()
    sel_MLE = np.zeros(nactive)

    for j in range(nactive):
        sel_MLE[j] = ci.approx_MLE_solver(j, step=1, nstep=150)[0]

    return sel_MLE

X, y, beta, nonzero, sigma = gaussian_instance(n=100, p=200, s=5, rho=0., signal=3., sigma=1.)
true_mean = X.dot(beta)
test = test_approximate_MLE(X,
                            y,
                            true_mean,
                            sigma,
                            seed_n = 0,
                            lam_frac = 1.,
                            loss='gaussian')
print(test)