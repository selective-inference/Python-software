from __future__ import print_function
import time
import random
import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.bayesian.cisEQTLS.inference_per_gene import selection_probability_variants, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from scipy.stats import norm as normal
from selection.bayesian.cisEQTLS.Simes_selection import BH_q


def one_trial(outputfile, X = None, y=None, seed_n = 19, pgenes= 0.8 , bh_level=0.1, method="theoretical"):

    if X is None and y is None:
        random.seed(seed_n)
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=7000, s=10, sigma=1, rho=0, snr=5.)

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z, method="theoretical")
    lam, epsilon, active, betaE, cube, initial_soln = sel

    if sel is not None:
        lagrange = lam * np.ones(p)
        active_sign = np.sign(betaE)
        nactive = active.sum()

        feasible_point = np.fabs(betaE)

        noise_variance = 1.

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

        samples = inf.posterior_samples()

        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        selective_mean = np.mean(samples, axis=0)

        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)
        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

        nerr = 0.

        active_set = [i for i in range(p) if active[i]]
        active_ind = np.zeros(p)
        active_ind[active_set] = 1

        list_results = []
        list_results.append(true_beta)
        list_results.append(active_ind)

        ad_lower_credible = np.zeros(p)
        ad_upper_credible = np.zeros(p)

        unad_lower_credible = np.zeros(p)
        unad_upper_credible = np.zeros(p)

        ad_mean = np.zeros(p)
        unad_mean = np.zeros(p)

        if nactive > 1:
            try:
                for l in range(nactive):
                    ad_lower_credible[active_set[l]] = adjusted_intervals[0, l]
                    ad_upper_credible[active_set[l]] = adjusted_intervals[1, l]
                    unad_lower_credible[active_set[l]] = unadjusted_intervals[0, l]
                    unad_upper_credible[active_set[l]] = unadjusted_intervals[1, l]
                    ad_mean[active_set[l]] = selective_mean[l]
                    unad_mean[active_set[l]] = post_mean[l]

            except ValueError:
                nerr += 1
                print('ignore iteration raising ValueError')

            ngrid = 1000
            quantiles = np.zeros((ngrid, nactive))
            for i in range(ngrid):
                quantiles[i, :] = np.percentile(samples, (i * 100.) / ngrid, axis=0)

            index_grid = np.argmin(np.abs(quantiles - np.zeros((ngrid, nactive))), axis=0)
            p_value = 2 * np.minimum(np.true_divide(index_grid, ngrid), 1. - np.true_divide(index_grid, ngrid))
            bh_level_ad = bh_level * pgenes
            p_BH = BH_q(p_value, bh_level_ad)

            D_BH = np.zeros(p)

            if p_BH is not None:
                for indx in p_BH[1]:
                    D_BH[active_set[indx]] = 1

            list_results.append(D_BH)

            list_results.append(ad_lower_credible)
            list_results.append(ad_upper_credible)
            list_results.append(unad_lower_credible)
            list_results.append(unad_upper_credible)

            list_results.append(ad_mean)
            list_results.append(unad_mean)

            with open(outputfile, "w") as output:
                for val in range(p):
                    output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(true_beta[val],
                                                                               active_ind[val],
                                                                               D_BH[val],
                                                                               ad_lower_credible[val],
                                                                               ad_upper_credible[val],
                                                                               unad_lower_credible[val],
                                                                               unad_upper_credible[val],
                                                                               ad_mean[val],
                                                                               unad_mean[val]))

            return list_results

one_trial("/Users/snigdhapanigrahi/Results_cisEQTLS/output.txt")