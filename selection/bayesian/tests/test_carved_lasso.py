from __future__ import print_function
import numpy as np
import regreg.api as rr

from ...tests.instance import logistic_instance, gaussian_instance
from ...tests.flags import SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue 

from ..par_carved_reduced import selection_probability_carved, sel_inf_carved
from ..estimator import M_estimator_approx_carved

def carved_lasso_trial(X,
                       y,
                       beta,
                       sigma,
                       lam,
                       estimation='parametric',
                       ndraw=1000,
                       burnin=100):
    n, p = X.shape

    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

    total_size = loss.saturated_loss.shape[0]
    subsample_size = int(0.8 * total_size)

    M_est = M_estimator_approx_carved(loss, epsilon, subsample_size, penalty, estimation)

    M_est.solve_approx()
    active = M_est._overall
    nactive = M_est.nactive

    if nactive >= 1:
        prior_variance = 1000.
        noise_variance = sigma ** 2
        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

        print("observed data", post_mean)

        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])
        grad_lasso = sel_inf_carved(M_est, prior_variance)
        samples = grad_lasso.posterior_samples(ndraw=ndraw, burnin=burnin)
        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        selective_mean = np.mean(samples, axis=0)

        coverage_ad = np.zeros(nactive)
        coverage_unad = np.zeros(nactive)
        ad_length = np.zeros(nactive)
        unad_length = np.zeros(nactive)

        true_val = np.zeros(nactive)
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
        bayes_risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive

        return np.vstack([sel_cov, naive_cov, ad_len, unad_len, bayes_risk_ad, bayes_risk_unad])

    else:
        return np.vstack([0.,0.,0.,0.,0.,0.])

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_carved_lasso(ndraw=1000, burnin=100):
    ### set parameters
    n = 1000
    p = 100
    s = 20
    snr = 7.

    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0, signal=snr)
    lam = 0.8 * np.mean(np.fabs(X.T.dot(np.random.standard_normal((n, 2000)))).max(0)) * sigma
    lasso = carved_lasso_trial(X,
                               y,
                               beta,
                               sigma,
                               lam,
                               ndraw=ndraw,
                               burnin=burnin)


    if lasso is not None:
        ad_cov += lasso[0,0]
        unad_cov += lasso[1,0]
        ad_len += lasso[2, 0]
        unad_len += lasso[3, 0]
        ad_risk += lasso[4, 0]
        unad_risk += lasso[5, 0]
        print("\n")
        print("\n")
        print("adjusted and unadjusted coverage", ad_cov, unad_cov)
        print("adjusted and unadjusted lengths", ad_len, unad_len)

