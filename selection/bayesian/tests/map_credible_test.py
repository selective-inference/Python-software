from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection
from selection.bayesian.selection_probability_rr import cube_subproblem, cube_gradient, cube_barrier, \
    selection_probability_objective, cube_subproblem_scaled, cube_gradient_scaled, cube_barrier_scaled, \
    cube_subproblem_scaled
from selection.randomized.api import randomization
from selection.bayesian.selection_probability import selection_probability_methods
from selection.bayesian.dual_scipy import dual_selection_probability_func
from selection.bayesian.inference_rr import sel_prob_gradient_map, selective_map_credible


def test_inf_regreg():
    n = 50
    p = 5
    s = 3
    snr = 5

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, true_beta, active, active.sum())
    noise_variance = 1.
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    primal_feasible = np.fabs(betaE)
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
    lagrange = lam * np.ones(p)
    generative_X = X_1[:, active]
    prior_variance = 1000.

    inf_rr = selective_map_credible(y,
                                    X_1,
                                    primal_feasible,
                                    dual_feasible,
                                    active,
                                    active_signs,
                                    lagrange,
                                    generative_X,
                                    noise_variance,
                                    prior_variance,
                                    randomization.isotropic_gaussian((p,), tau),
                                    epsilon)

    map = inf_rr.map_solve_2(nstep = 100)[::-1]

    print ("gradient at map", -inf_rr.smooth_objective(map[1], mode='grad'))
    print ("map objective, map", map[0], map[1])
    toc = time.time()
    samples = inf_rr.posterior_samples()
    tic = time.time()
    print('sampling time', tic - toc)
    return samples

#test_inf_regreg()
post_samples = test_inf_regreg()
print(np.percentile(post_samples, 5, axis=0), np.percentile(post_samples, 95, axis=0))


