from __future__ import print_function
import time

import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.ms_lasso_2stage import selection_probability_objective_ms_lasso, sel_prob_gradient_map_ms_lasso,\
    selective_map_credible_ms_lasso

def sel_prob_ms_lasso():
    n = 100
    p = 200
    s = 10
    snr = 5

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    Z_stats = X_1.T.dot(y)
    randomized_Z_stats = np.true_divide(Z_stats, noise_variance) + random_Z

    active_1 = np.zeros(p, bool)
    active_1[np.fabs(randomized_Z_stats) > 1.65] = 1
    active_signs_1 = np.sign(randomized_Z_stats[active_1])
    nactive_1 = active_1.sum()
    print("active_1",active_1, nactive_1)

    threshold = 1.65 * np.ones(p)

    X_step2 = X_1[:, active_1]
    random_Z_2 = np.random.standard_normal(nactive_1)
    sel = selection(X_step2, y, random_Z_2)
    lam, epsilon, active_2, betaE, cube, initial_soln = sel
    noise_variance = 1.
    lagrange = lam * np.ones(nactive_1)
    nactive_2 = betaE.shape[0]
    print("active_2", active_2, nactive_2)
    active_signs_2 = np.sign(betaE)

    #getting the active indices
    active = np.zeros(p, bool)
    indices_stage2 = np.where(active_1 == 1)[0]
    active[indices_stage2[active_2]] = 1
    print("the active indices after two stages of screening", active, active.sum())

    primal_feasible_1 = np.fabs(randomized_Z_stats[active_1])
    primal_feasible_2 = np.fabs(betaE)
    feasible_point = np.append(primal_feasible_1, primal_feasible_2)

    randomizer = randomization.isotropic_gaussian((p,), 1.)
    #parameter = np.random.standard_normal(nactive_2)

    #mean = X_step2[:, active_2].dot(parameter)

    #test_point = np.append(np.append(np.random.uniform(low=-2.0, high=2.0, size=n), primal_feasible_1),
    #                       primal_feasible_2)

    #ms_lasso = selection_probability_objective_ms_lasso(X_1,
    #                                                       feasible_point, #in R^{|E|_1 + |E|_2}
    #                                                       active_1, #the active set chosen by randomized marginal screening
    #                                                       active_2, #the active set chosen by randomized lasso
    #                                                       active_signs_1, #the set of signs of active coordinates chosen by ms
    #                                                       active_signs_2, #the set of signs of active coordinates chosen by lasso
    #                                                       lagrange, #in R^p
    #                                                       threshold, #in R^p
    #                                                       mean, # in R^n
    #                                                       noise_variance,
    #                                                       randomizer,
    #                                                       epsilon)

    #sel_prob_ms_lasso = ms_lasso.minimize2(nstep=100)[::-1]
    #print("selection prob and minimizer- fs", sel_prob_ms_lasso[0], sel_prob_ms_lasso[1])

    generative_X = X_step2[:, active_2]
    prior_variance = 100.

    Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
    post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
    post_var = (prior_variance * np.identity(nactive_2)) - ((prior_variance ** 2) * (generative_X.T.dot(Q).dot(generative_X)))
    unadjusted_intervals = np.vstack([post_mean - 1.65 * (post_var.diagonal()), post_mean + 1.65 * (post_var.diagonal())])
    unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])

    grad_map = sel_prob_gradient_map_ms_lasso(X_1,
                                              feasible_point,  # in R^{|E|_1 + |E|_2}
                                              active_1,  # the active set chosen by randomized marginal screening
                                              active_2,  # the active set chosen by randomized lasso
                                              active_signs_1,  # the set of signs of active coordinates chosen by ms
                                              active_signs_2,  # the set of signs of active coordinates chosen by lasso
                                              lagrange,  # in R^p
                                              threshold,  # in R^p
                                              generative_X,  # in R^{p}\times R^{n}
                                              noise_variance,
                                              randomizer,
                                              epsilon)

    ms = selective_map_credible_ms_lasso(y,
                                         grad_map,
                                         prior_variance)

    sel_MAP = ms.map_solve(nstep=100)[::-1]

    print("selective MAP- ms_lasso_screening", sel_MAP[1])

    toc = time.time()
    samples = ms.posterior_samples()
    tic = time.time()
    print('sampling time', tic - toc)

    sel_mean = np.mean(samples, axis=0)
    adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
    adjusted_intervals = np.vstack([sel_mean, np.vstack([sel_MAP[1], adjusted_intervals])])

    active_set = [i for i in range(p) if active[i]]
    print("active set", active_set)
    print("adjusted inference", adjusted_intervals)
    print("unadjusted inference", unadjusted_intervals)

sel_prob_ms_lasso()










