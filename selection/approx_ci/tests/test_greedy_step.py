from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_approx_greedy_step import neg_log_cube_probability_fs, approximate_conditional_prob_fs, \
    approximate_conditional_density
from selection.approx_ci.estimator_approx import greedy_score_step_approx

def test_approximate_ci(n=100,
                        p=10,
                        s=0,
                        snr=5,
                        rho=0.1,
                        lam_frac = 1.,
                        loss='gaussian',
                        randomizer='gaussian'):

    from selection.api import randomization

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        loss = rr.glm.gaussian(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    if randomizer == 'gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer == 'laplace':
        randomization = randomization.laplace((p,), scale=1.)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    # active_bool = np.zeros(p, np.bool)
    # active_bool[range(3)] = 1
    # inactive_bool = ~active_bool

    GS = greedy_score_step_approx(loss,
                                  penalty,
                                  np.zeros(p, dtype=bool),
                                  np.ones(p, dtype=bool),
                                  randomization,
                                  randomizer)

    GS.solve_approx()
    active = GS._overall
    print("nactive", active.sum())

    ci = approximate_conditional_density(GS)
    ci.solve_approx()

    active_set = np.asarray([i for i in range(p) if active[i]])
    true_support = np.asarray([i for i in range(p) if i < s])
    nactive = np.sum(active)
    print("active set, true_support", active_set, true_support)
    true_vec = beta[active]
    print("true coefficients", true_vec)

    if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:

        ci_active = np.zeros((nactive, 2))
        covered = np.zeros(nactive, np.bool)
        ci_length = np.zeros(nactive)
        pivots = np.zeros(nactive)

        toc = time.time()

        for j in range(nactive):
            ci_active[j, :] = np.array(ci.approximate_ci(j))
            if (ci_active[j, 0] <= true_vec[j]) and (ci_active[j, 1] >= true_vec[j]):
                covered[j] = 1
            ci_length[j] = ci_active[j, 1] - ci_active[j, 0]
            # print(ci_active[j, :])
            pivots[j] = ci.approximate_pvalue(j, true_vec[j])

        print("confidence intervals", ci_active)
        tic = time.time()
        print('ci time now', tic - toc)


test_approximate_ci()
