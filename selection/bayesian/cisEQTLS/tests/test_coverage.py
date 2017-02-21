from __future__ import print_function
import time

import os, numpy as np, pandas, statsmodels.api as sm
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.bayesian.cisEQTLS.inference_2sels import selection_probability_genes_variants, \
    sel_prob_gradient_map_simes_lasso, selective_inf_simes_lasso
from scipy.stats import norm as normal

def test_coverage():
    n = 350
    p = 5000
    s = 0
    snr = 0.

    X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

    n, p = X.shape

    alpha = 0.05

    sel_simes = simes_selection(X, y, alpha=0.05, randomizer='gaussian')

    if sel_simes is not None:

        index = sel_simes[0]

        t_0 = sel_simes[2]

        J = sel_simes[1]

        T_sign = sel_simes[3]*np.ones(1)

        T_stats = sel_simes[4]*np.ones(1)

        if t_0 == 0:
            threshold = normal.ppf(1.- alpha/(2.*p))*np.ones(1)

        else:
            J_card = J.shape[0]
            threshold = np.zeros(J_card+1)
            threshold[:J_card] = normal.ppf(1. - (alpha/(2.*p))*(np.arange(J_card)+1.))
            threshold[J_card] = normal.ppf(1. - (alpha/(2.*p))*t_0)

        random_Z = np.random.standard_normal(p)
        sel = selection(X, y, random_Z)
        lam, epsilon, active, betaE, cube, initial_soln = sel

        if sel is not None:

            lagrange = lam * np.ones(p)
            active_sign = np.sign(betaE)
            nactive = active.sum()
            print("number of selected variables by Lasso", nactive)

            feasible_point = np.append(1, np.fabs(betaE))

            noise_variance = 1.

            randomizer = randomization.isotropic_gaussian((p,), 1.)

            generative_X = X[:, active]
            prior_variance = 1000.

            grad_map = sel_prob_gradient_map_simes_lasso(X,
                                                         feasible_point,
                                                         index,
                                                         J,
                                                         active,
                                                         T_sign,
                                                         active_sign,
                                                         lagrange,
                                                         threshold,
                                                         generative_X,
                                                         noise_variance,
                                                         randomizer,
                                                         epsilon)

            inf = selective_inf_simes_lasso(y, grad_map, prior_variance)

            samples = inf.posterior_samples()

            adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

            Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
            post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
            post_var = prior_variance * np.identity(nactive) - ((prior_variance ** 2) * (generative_X.T.dot(Q).dot(generative_X)))
            unadjusted_intervals = np.vstack([post_mean - 1.65 * (post_var.diagonal()), post_mean + 1.65 * (post_var.diagonal())])

            coverage_ad = np.zeros(p)
            coverage_unad = np.zeros(p)
            nsel = np.zeros(p)
            nerr = 0

            true_val = true_beta[active]
            active_set = [i for i in range(p) if active[i]]

            if nactive > 1:
                try:
                    for l in range(nactive):
                        nsel[active_set[l]] += 1
                        if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                            coverage_ad[active_set[l]] += 1
                        if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                            coverage_unad[active_set[l]] += 1
                    #print('coverage adjusted so far', coverage_ad)
                    #print('coverage unadjusted so far', coverage_unad)

                except ValueError:
                    nerr += 1
                    print('ignore iteration raising ValueError')

                return coverage_ad.sum() / nactive, coverage_unad.sum() / nactive, nsel, nerr

            else:
                return 0., 0., nsel, 0.




cov_ad = 0.
cov_unad = 0.
niter = 10
for i in range(niter):

    cov = test_coverage()
    if np.any(cov[2] > 0.):

        cov_ad += cov[0]
        cov_unad += cov[1]
        print('coverage adjusted so far', cov_ad)
        print('coverage unadjusted so far',cov_unad)
        print("\n")
        print("iteration completed", i)
