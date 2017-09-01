from __future__ import print_function
import numpy as np
import regreg.api as rr
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_approx_greedy_step import (greedy_score_step_map,
                                                       approximate_conditional_density)

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

def test_approximate_inference(X,
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
        loss = rr.glm.gaussian(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    elif loss == "logistic":
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

        for j in range(nactive):
            ci_active[j, :] = np.array(ci.approximate_ci(j))
            if (ci_active[j, 0] <= true_vec[j]) and (ci_active[j, 1] >= true_vec[j]):
                covered[j] = 1
            ci_length[j] = ci_active[j, 1] - ci_active[j, 0]
            # print(ci_active[j, :])
            pivots[j] = ci.approximate_pvalue(j, true_vec[j])

        print("confidence intervals", ci_active)
        print('ci time now', tic - toc)


test_approximate_ci()
