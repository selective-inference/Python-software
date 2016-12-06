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
from selection.bayesian.inference_fs import sel_prob_gradient_map_fs, selective_map_credible_fs
from selection.bayesian.inference_ms import sel_prob_gradient_map_ms, selective_map_credible_ms

def test_mle_lasso():
    n = 60
    p = 30
    s = 10
    snr = 6

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, true_beta, active, active.sum())

    if active[:s].sum() == s :
        noise_variance = 1.
        nactive = betaE.shape[0]
        active_signs = np.sign(betaE)
        tau = 1  # randomization_variance
        primal_feasible = np.fabs(betaE)
        dual_feasible = np.ones(p)
        dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
        lagrange = lam * np.ones(p)
        generative_X = X_1[:,active]
        prior_variance = 100000.

        Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
        mle_unadjusted = prior_variance * ((generative_X.T.dot(Q)).dot(y))

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

        map = inf_rr.map_solve_2(nstep=200)[::-1]
        print("selective mle", map[1])
        print("unadjusted mle", mle_unadjusted)
        print("selective mse", ((map[1]-true_beta[active])**2).sum()/nactive)
        print("usual mse", ((mle_unadjusted - true_beta[active]) ** 2).sum()/nactive)

        return map[1], mle_unadjusted

test_mle_lasso()


