from __future__ import print_function
import numpy as np
import time
import regreg.api as rr

from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_via_approx_density import approximate_conditional_density
from selection.approx_ci.estimator_approx import M_estimator_approx

def test_approximate_mle(n=100,
                         p=10,
                         s=3,
                         snr=5,
                         rho=0.1,
                         lam_frac = 1.,
                         loss='gaussian',
                         randomizer='gaussian'):

    from selection.api import randomization

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    if randomizer == 'gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer == 'laplace':
        randomization = randomization.laplace((p,), scale=1.)

    M_est = M_estimator_approx(loss, epsilon, penalty, randomization, randomizer)
    M_est.solve_approx()

    inf = approximate_conditional_density(M_est)
    inf.solve_approx()

    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])

    true_support = np.asarray([i for i in range(p) if i < s])

    nactive = np.sum(active)

    print("active set, true_support", active_set, true_support)

    true_vec = beta[active]

    print("true coefficients", true_vec)

    if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:

        mle_active = np.zeros(nactive)

        for j in range(nactive):
            mle_active[j] = inf.approx_MLE_solver(j, nstep=100)[0]

        print("mle for target", mle_active)

test_approximate_mle()

