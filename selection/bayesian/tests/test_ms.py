from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.bayesian.selection_probability_rr import cube_gradient_scaled, cube_barrier_scaled, cube_subproblem_scaled
from selection.bayesian.marginal_screening import selection_probability_objective_ms
from selection.randomized.api import randomization


def ms_primal_test():
    n = 50
    p = 10
    s = 5
    snr = 3

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    print(true_beta, nonzero, noise_variance)
    random_Z = np.random.standard_normal(p)
    w, v = np.linalg.eig(X_1.T.dot(X_1))
    var_half_inv = (v.T.dot(np.diag(np.power(w,-0.5)))).dot(v)
    Z_stats = var_half_inv.dot(X_1.T.dot(y))
    random_Z = np.true_divide(Z_stats,noise_variance)  + random_Z


    active = np.zeros(p, bool)
    active[np.fabs(random_Z)>1.65] = 1
    active_signs = np.sign(random_Z[active])
    nactive = active.sum()

    feasible_point = np.ones(nactive)
    tau = 1 #randomization_variance

    parameter = np.random.standard_normal(nactive)
    mean = (var_half_inv.dot(X_1.T)).dot(X_1[:, active].dot(parameter))
    randomizer = randomization.isotropic_gaussian((p,), 1.)
    threshold = 1.65 * np.ones(p)

    ms = selection_probability_objective_ms(feasible_point,
                                            active,
                                            active_signs,
                                            threshold, # a vector in R^p
                                            mean,  # in R^p
                                            noise_variance,
                                            randomizer)


ms_primal_test()