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
    #print("here",glm.shape)
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    glm = M_est.observed_score_state[:nactive]

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

    selective_mean = np.mean(samples, axis=0)

    true_val = np.zeros(nactive)

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
    bayes_risk_ad = np.power(selective_mean - true_val, 2.).sum() / nactive
    bayes_risk_unad = np.power(glm - true_val, 2.).sum() / nactive

    return np.vstack([sel_cov, naive_cov, ad_len, unad_len, bayes_risk_ad, bayes_risk_unad])


if __name__ == "__main__":
    ### set parameters
    n = 500
    p = 100
    s = 0
    snr = 0.

    niter = 3
    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.

    for i in range(niter):

         ### GENERATE X, Y BASED ON SEED
         np.random.seed(i+68)  # ensures different X and y
         #X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0., snr=snr)
         # lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
         X, y, beta, nonzero = logistic_instance(n=n, p=p, s=s, rho=0., snr=snr)
         lam = 1.5 * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
         sigma = 1.

         ### RUN LASSO AND TEST
         lasso = randomized_lasso_trial(X,
                                        y,
                                        beta,
                                        sigma,
                                        lam)

         if lasso is not None:
             ad_cov += lasso[0,0]
             unad_cov += lasso[1,0]
             ad_len += lasso[2,0]
             unad_len += lasso[3,0]
             ad_risk += lasso[4, 0]
             unad_risk += lasso[5, 0]
             print("\n")
             print("iteration completed", i)
             print("\n")
             print("adjusted and unadjusted coverage", ad_cov, unad_cov)
             print("adjusted and unadjusted lengths", ad_len, unad_len)
             print("adjusted and unadjusted risks", ad_risk, unad_risk)


    print("adjusted and unadjusted coverage",ad_cov, unad_cov)
    print("adjusted and unadjusted lengths", ad_len, unad_len)
    print("adjusted and unadjusted risks", ad_risk, unad_risk)

# if __name__ == "__main__":
#
#
#     seedn = int(sys.argv[1])
#     outdir = sys.argv[2]
#
#     outfile = os.path.join(outdir, "list_result_" + str(seedn) + ".txt")
#
#     ### set parameters
#     n = 1000
#     p = 200
#     s = 0
#     snr = 0.
#
#     ### GENERATE X, Y BASED ON SEED
#     np.random.seed(seedn+50)  # ensures different X and y
#     X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)
#     lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
#
#     ### RUN LASSO AND TEST
#     lasso = randomized_lasso_trial(X,
#                                    y,
#                                    beta,
#                                    sigma,
#                                    lam)
#
#     np.savetxt(outfile, lasso)
