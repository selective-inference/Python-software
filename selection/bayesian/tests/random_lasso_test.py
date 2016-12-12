from __future__ import print_function
import time

import numpy as np
import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.randomized.api import randomization
from selection.bayesian.paired_bootstrap import pairs_bootstrap_glm, bootstrap_cov
from selection.bayesian.randomX_lasso_primal import selection_probability_objective_randomX
from selection.bayesian.inference_randomX_lasso import sel_prob_gradient_map_randomX, selective_map_credible_randomX

#test for selection probability for random X
def test_approximation_randomlasso():
    n = 200
    p = 20
    s = 5
    snr = 3

    sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()
    random_Z = np.random.standard_normal(p)
    sel = selection(X_1, y, random_Z, randomization_scale=1, sigma=None, lam=None)
    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(true_beta, active)

    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    bootstrap_score = pairs_bootstrap_glm(rr.glm.gaussian(X_1,y), active, beta_full=None, inactive = ~active)[0]
    sampler = lambda: np.random.choice(n, size=(n,),replace = True)
    cov = bootstrap_cov(sampler, bootstrap_score)

    X_active = X_1[:, active]
    X_inactive = X_1[:,~active]
    X_gen_inv = np.linalg.pinv(X_active)
    X_projection = X_active.dot(X_gen_inv)
    parameter = np.random.standard_normal(nactive)
    lagrange = lam * np.ones(p)

    #selected model mean
    generative_mean = X_active.dot(parameter)
    mean_suff = X_gen_inv.dot(generative_mean)
    mean_nuisance = ((X_inactive.T).dot((np.identity(n)- X_projection))).dot(generative_mean)
    mean = np.append(mean_suff, mean_nuisance)
    print("means", mean_suff, mean_nuisance)

    sel_prob_regreg = selection_probability_objective_randomX(X_1,
                                                              np.fabs(betaE),
                                                              active,
                                                              active_signs,
                                                              lagrange,
                                                              mean,
                                                              cov,
                                                              noise_variance,
                                                              randomization.isotropic_gaussian((p,), 1),
                                                              epsilon)

    toc = time.time()
    regreg = sel_prob_regreg.minimize2(nstep=100)[::-1]
    tic = time.time()
    print('computation time', tic - toc)
    print('selection prob', regreg[0])
    print('minimizer', regreg[1])

#test_approximation_randomlasso()

#test for inference in random_X
def test_inference_randomlasso():
    n = 200
    p = 20
    s = 5
    snr = 3

    sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()
    random_Z = np.random.standard_normal(p)
    sel = selection(X_1, y, random_Z, randomization_scale=1, sigma=None, lam=None)
    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(true_beta, active)

    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    bootstrap_score = pairs_bootstrap_glm(rr.glm.gaussian(X_1,y), active, beta_full=None, inactive = ~active)[0]
    sampler = lambda: np.random.choice(n, size=(n,),replace = True)
    cov = bootstrap_cov(sampler, bootstrap_score)
    print("bootstrapped covariance", cov)

    primal_feasible = np.fabs(betaE)
    lagrange = lam * np.ones(p)
    generative_X = X_1[:, active]
    prior_variance = 1000.

    Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
    post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
    post_var = prior_variance * np.identity(nactive) - (
    (prior_variance ** 2) * (generative_X.T.dot(Q).dot(generative_X)))
    unadjusted_intervals = np.vstack([post_mean - 1.65 * (post_var.diagonal()), post_mean
                                      + 1.65 * (post_var.diagonal())])

    inf_rr = selective_map_credible_randomX(y,
                                            X_1,
                                            primal_feasible,
                                            active,
                                            active_signs,
                                            lagrange,
                                            cov,
                                            generative_X,
                                            noise_variance,
                                            prior_variance,
                                            randomization.isotropic_gaussian((p,), tau),
                                            epsilon)

    map = inf_rr.map_solve_2(nstep=100)[::-1]

    toc = time.time()
    samples = inf_rr.posterior_samples()
    tic = time.time()
    print('sampling time', tic - toc)
    adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
    print("selective intervals", adjusted_intervals)
    print("unadjusted intervals", unadjusted_intervals)
    print("selective map", map[1])
    print("unadjusted map", post_mean)
    print("active", active)

    adjusted_intervals = np.vstack([map[1], adjusted_intervals])
    return(adjusted_intervals)


test_inference_randomlasso()





