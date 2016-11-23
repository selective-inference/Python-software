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

def test_inf_regreg():
    n = 100
    p = 30
    s = 5
    snr = 3

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

    Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
    post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
    post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
    unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])
    unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])
    #print(np.vstack([post_mean, unadjusted_intervals]))

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
    print("selective map", map[1])

    #print ("gradient at map", -inf_rr.smooth_objective(map[1], mode='grad'))
    #print ("map objective, map", map[0], map[1])
    toc = time.time()
    samples = inf_rr.posterior_samples()
    tic = time.time()
    print('sampling time', tic - toc)
    adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
    adjusted_intervals = np.vstack([map[1], adjusted_intervals])
    print(active)
    print("selective map and intervals", adjusted_intervals)
    print("usual posterior based map & intervals", unadjusted_intervals)
    return np.vstack([unadjusted_intervals, adjusted_intervals])

#intervals = test_inf_regreg()
#np.savetxt('credible_randomized.txt', intervals)
#post_samples = test_inf_regreg()
#adjusted_intervals = np.vstack([np.percentile(post_samples, 5, axis=0), np.percentile(post_samples, 95, axis=0)])
#print(adjusted_intervals)
#intervals = np.vstack([unadjusted_intervals, adjusted_intervals])




def test_inf_fs():
    n = 50
    p = 5
    s = 3
    snr = 5

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    print(true_beta, nonzero, noise_variance)
    random_Z = np.random.standard_normal(p)
    random_obs = X_1.T.dot(y) + random_Z
    active_index = np.argmax(random_obs)
    active = np.zeros(p, bool)
    active[active_index] = 1
    active_sign = np.sign(random_obs[active_index])
    nactive = 1

    primal_feasible = np.fabs(random_obs[active_index])
    tau = 1  # randomization_variance

    parameter = np.random.standard_normal(nactive)
    mean = X_1[:, active].dot(parameter)

    generative_X = X_1[:, active]
    prior_variance = 1000.

    inf_rr = selective_map_credible_fs(y,
                                       X_1,
                                       primal_feasible,
                                       active,
                                       active_sign,
                                       generative_X,
                                       noise_variance,
                                       prior_variance,
                                       randomization.isotropic_gaussian((p,), tau))

    map = inf_rr.map_solve_2(nstep = 100)[::-1]

    print ("gradient at map", -inf_rr.smooth_objective(map[1], mode='grad'))
    print ("map objective, map", map[0], map[1])
    toc = time.time()
    samples = inf_rr.posterior_samples()
    tic = time.time()
    print('sampling time', tic - toc)
    return samples

#test_inf_fs()
#post_samples = test_inf_fs()
#print(np.percentile(post_samples, 5, axis=0), np.percentile(post_samples, 95, axis=0))

def test_inf_ms():
    n = 50
    p = 10
    s = 5
    snr = 5

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    print(true_beta, nonzero, noise_variance)
    random_Z = np.random.standard_normal(p)
    randomized_Z_stats = np.true_divide(X_1.T.dot(y), noise_variance) + random_Z

    active = np.zeros(p, bool)
    active[np.fabs(randomized_Z_stats) > 1.65] = 1
    active_signs = np.sign(randomized_Z_stats[active])
    nactive = active.sum()
    randomizer = randomization.isotropic_gaussian((p,), 1.)
    threshold = 1.65 * np.ones(p)

    generative_X = X_1[:, active]
    prior_variance = 1000.

    inf_rr = selective_map_credible_ms(y,
                                       X_1,
                                       active,
                                       active_signs,
                                       threshold,
                                       generative_X,
                                       noise_variance,
                                       prior_variance,
                                       randomizer)

    map = inf_rr.map_solve_2(nstep = 100)[::-1]

    print ("gradient at map", -inf_rr.smooth_objective(map[1], mode='grad'))
    print ("map objective, map", map[0], map[1])
    toc = time.time()
    samples = inf_rr.posterior_samples()
    tic = time.time()
    print('sampling time', tic - toc)
    return samples

#test_inf_ms()