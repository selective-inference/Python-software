from __future__ import print_function
import numpy as np

from ...randomized.api import randomization
from ..initial_soln import selection, instance

from ..dual_lasso import (selection_probability_lasso_dual,
                          sel_prob_gradient_map_lasso,
                          selective_inf_lasso)

from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.decorators import (set_sampling_params_iftrue,
                                 set_seed_iftrue)

def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma,
                           ndraw=1000,
                           burnin=100):

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z)
    lam, epsilon, active, betaE, cube, initial_soln = sel

    if sel is not None:

        lagrange = lam * np.ones(p)
        active_sign = np.sign(betaE)
        nactive = active.sum()
        print("number of selected variables by Lasso", nactive)

        feasible_point = np.ones(p)
        feasible_point[:nactive] = -np.fabs(betaE)

        noise_variance = sigma ** 2

        randomizer = randomization.isotropic_gaussian((p,), 1.)

        generative_X = X[:, active]
        prior_variance = 1000.

        grad_map = sel_prob_gradient_map_lasso(X,
                                               feasible_point,
                                               active,
                                               active_sign,
                                               lagrange,
                                               generative_X,
                                               noise_variance,
                                               randomizer,
                                               epsilon)

        inf = selective_inf_lasso(y, grad_map, prior_variance)

        samples = inf.posterior_samples(ndraw=ndraw, burnin=burnin)

        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        selective_mean = np.mean(samples, axis=0)

        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

        print("observed data", post_mean)

        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

        coverage_ad = np.zeros(nactive)
        coverage_unad = np.zeros(nactive)
        ad_length = np.zeros(nactive)
        unad_length = np.zeros(nactive)

        true_val = projection_active.T.dot(X.dot(beta))

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
        risk_ad = np.power(selective_mean - true_val, 2.).sum() / nactive
        risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive

        return np.vstack([sel_cov, naive_cov, ad_len, unad_len, risk_ad, risk_unad])

    else:
        return None

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=20)
def test_dual_lasso(ndraw=1000, burnin=100):
    ### set parameters
    n = 300
    p = 100
    s = 10
    snr = 7.

    sample = instance(n=n, p=p, s=s, sigma=1., rho=0, signal=snr)

    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.

    X, y, beta, nonzero, sigma = sample.generate_response()

    ### RUN LASSO AND TEST
    lasso = randomized_lasso_trial(X,
                                   y,
                                   beta,
                                   sigma,
                                   ndraw=ndraw,
                                   burnin=burnin)

    if lasso is not None:
        ad_cov += lasso[0,0]
        unad_cov += lasso[1,0]
        ad_len += lasso[2, 0]
        unad_len += lasso[3, 0]
        print("\n")
        print("adjusted and unadjusted coverage", ad_cov, unad_cov)
        print("\n")
        print("adjusted and unadjusted lengths", ad_len, unad_len)

