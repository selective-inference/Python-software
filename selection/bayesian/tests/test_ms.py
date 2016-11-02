from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.bayesian.selection_probability_rr import cube_gradient_scaled, cube_barrier_scaled, cube_subproblem_scaled
from selection.bayesian.marginal_screening import selection_probability_objective_ms, dual_selection_probability_ms
from selection.randomized.api import randomization


def ms_primal_dual_test():
    n = 30
    p = 10
    s = 5
    snr = 3

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    print(true_beta, noise_variance)

    random_Z = np.random.standard_normal(p)
    #w, v = np.linalg.eig(X_1.T.dot(X_1))
    #var_half_inv = (v.T.dot(np.diag(np.power(w, -0.5)))).dot(v)
    Z_stats = X_1.T.dot(y)
    randomized_Z_stats = np.true_divide(Z_stats, noise_variance) + random_Z

    active = np.zeros(p, bool)
    active[np.fabs(randomized_Z_stats)>1.65] = 1
    active_signs = np.sign(randomized_Z_stats[active])
    nactive = active.sum()
    tau = 1 #randomization_variance

    X = np.hstack([X_1[:, active], X_1[:, ~active]])
    w_0, v_0 = np.linalg.eig(X.T.dot(X))
    var_half_inv_0 = (v_0.T.dot(np.diag(np.power(w_0, -0.5)))).dot(v_0)

    parameter = np.random.standard_normal(nactive)
    mean = (var_half_inv_0.dot(X.T)).dot(X_1[:, active].dot(parameter))

    randomizer = randomization.isotropic_gaussian((p,), 1.)
    threshold = 1.65 * np.ones(p)

    ms_primal = selection_probability_objective_ms(active,
                                                   active_signs,
                                                   threshold, # a vector in R^p
                                                   mean,  # in R^p
                                                   noise_variance,
                                                   randomizer)

    ms_dual = dual_selection_probability_ms(active,
                                       active_signs,
                                       threshold,  # a vector in R^p
                                       mean,  # in R^p
                                       noise_variance,
                                       randomizer)

    #test = np.append(np.ones(p),np.fabs(np.random.standard_normal(nactive)))
    #print(ms.smooth_objective(test))

    toc = time.time()
    sel_prob_primal = ms_primal.minimize2(nstep=50)[::-1]
    tic = time.time()
    print('primal ms time', tic - toc)

    toc = time.time()
    sel_prob_dual = ms_dual.minimize2(nstep=60)[::-1]
    tic = time.time()
    print('dual ms time', tic - toc)

    print("selection prob and minimizer- ms", sel_prob_primal, sel_prob_dual)

ms_primal_dual_test()

