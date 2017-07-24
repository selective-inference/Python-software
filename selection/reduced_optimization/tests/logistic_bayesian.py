from __future__ import print_function
import sys
import os
import numpy as np
import time
import regreg.api as rr
from selection.reduced_optimization.initial_soln import selection
from selection.tests.instance import logistic_instance, gaussian_instance

#from selection.reduced_optimization.random_lasso_reduced import selection_probability_random_lasso, sel_inf_random_lasso
from selection.reduced_optimization.par_random_lasso_reduced import selection_probability_random_lasso, sel_inf_random_lasso
from selection.reduced_optimization.estimator import M_estimator_approx, M_estimator_approx_logistic
from selection.randomized.query import naive_confidence_intervals

def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma,
                           lam,
                           loss ='logistic',
                           randomizer='gaussian',
                           estimation='parametric'):

    from selection.api import randomization

    n, p = X.shape
    if loss == "gaussian":
        loss = rr.glm.gaussian(X, y)

    elif loss == "logistic":
        loss = rr.glm.logistic(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),weights=dict(zip(np.arange(p), W)), lagrange=1.)
    randomization = randomization.isotropic_gaussian((p,), scale=1.)

    M_est = M_estimator_approx_logistic(loss, epsilon, penalty, randomization, randomizer, estimation)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    prior_variance = 100000.
    #generative_mean = np.zeros(p)
    #sel_split = selection_probability_random_lasso(M_est, generative_mean)
    #test_point = np.append(M_est.observed_score_state, np.abs(M_est.initial_soln[M_est._overall]))

    #print("gradient at test point", sel_split.smooth_objective(test_point, mode= "grad"))


    class target_class(object):
        def __init__(self, target_cov):
            self.target_cov = target_cov
            self.shape = target_cov.shape

    target = target_class(M_est.target_cov)
    unadjusted_intervals =(naive_confidence_intervals(target, M_est.target_observed)).T

    grad_lasso = sel_inf_random_lasso(M_est, prior_variance)
    samples = grad_lasso.posterior_samples()
    adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

    true_val = projection_active.T.dot(X.dot(beta))

    coverage_ad = np.zeros(nactive)
    coverage_unad = np.zeros(nactive)
    ad_length = np.zeros(nactive)
    unad_length = np.zeros(nactive)

    for l in range(nactive):
        if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
            coverage_ad[l] += 1
        ad_length[l] = adjusted_intervals[1, l] - adjusted_intervals[0, l]
        if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
            coverage_unad[l] += 1
        unad_length[l] = unadjusted_intervals[1, l] - unadjusted_intervals[0, l]

    sel_cov = coverage_ad.sum() / nactive
    naive_cov = coverage_unad.sum() / nactive
    ad_len = ad_length.sum() / nactive
    unad_len = unad_length.sum() / nactive

    return np.vstack([sel_cov, naive_cov, ad_len, unad_len])
