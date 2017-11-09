from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, selective_MLE

def test(n=100, p=50, s=5, signal=5., seed_n = 0, lam_frac=1., randomization_scale=1.):
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
    n, p = X.shape
    np.random.seed(seed_n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
    #randomizer = randomization.gaussian(np.identity(p))
    M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale)

    M_est.solve_approx()
    active = M_est._overall
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")

    solve_mle = selective_MLE(M_est)
    mle = np.zeros(nactive)
    for j in range(nactive):
        mle[j] = solve_mle.solve_UMVU(j)[0]

    return np.transpose(np.vstack([mle, M_est.target_observed]))

print(test())