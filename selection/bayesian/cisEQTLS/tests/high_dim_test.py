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


def one_trial(txtfile, n=350, p= 5000, s= 10, snr = 5., seed_n = 19, bh_level=0.1, method="theoretical"):

    random.seed(seed_n)

    sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

    X, y, true_beta, nonzero, noise_variance = sample.generate_response()

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z, method ="theoretical")
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
        M_1 = prior_variance *(X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance *((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)
        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

        coverage_ad = np.zeros(p)
        coverage_unad = np.zeros(p)
        nerr = 0.

        true_val = projection_active.T.dot(X.dot(true_beta))
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
                    if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                        coverage_ad[active_set[l]] += 1
                    if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                        coverage_unad[active_set[l]] += 1
                    ad_lower_credible[active_set[l]] = adjusted_intervals[0, l]
                    ad_upper_credible[active_set[l]] = adjusted_intervals[1, l]
                    unad_lower_credible[active_set[l]] = unadjusted_intervals[0, l]
                    unad_upper_credible[active_set[l]] = unadjusted_intervals[1, l]
                    ad_mean[active_set[l]] = selective_mean[l]
                    unad_mean[active_set[l]] = post_mean[l]

            except ValueError:
                nerr += 1
                print('ignore iteration raising ValueError')

            list_results.append(coverage_ad)
            list_results.append(coverage_unad)

            ngrid = 1000
            quantiles = np.zeros((ngrid, nactive))
            for i in range(ngrid):
                quantiles[i, :] = np.percentile(samples, (i * 100.) / ngrid, axis=0)

            index_grid = np.argmin(np.abs(quantiles - np.zeros((ngrid, nactive))), axis=0)
            p_value = 2 * np.minimum(np.true_divide(index_grid, ngrid), 1. - np.true_divide(index_grid, ngrid))
            p_BH = BH_q(p_value, bh_level)

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

            #txtfile = "/Users/snigdhapanigrahi/Results_cisEQTLS/output.txt"

            # Assuming res is a flat list
            with open(txtfile, "w") as output:
                for val in range(p):
                    output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(true_beta[val],
                                                                                       active_ind[val], coverage_ad[val],
                                                                                       coverage_unad[val],
                                                                                       D_BH[val], ad_lower_credible[val],
                                                                                       ad_upper_credible[val],
                                                                                       unad_lower_credible[val],
                                                                                       unad_upper_credible[val],
                                                                                       ad_mean[val],
                                                                                       unad_mean[val]))

            return list_results



R = one_trial("/Users/snigdhapanigrahi/Results_cisEQTLS/output.txt")
#print("true parameter",R[0])
#print("active indices",R[1])
#print("indices covered by adjusted",R[2])
#print("indices covered by unadjusted",R[3])
#print("indices declared significant after BH",R[4])
#print("adjusted lower bounds", R[5])
#print("adjusted upper bounds", R[6])
#print("unadjusted lower bounds", R[7])
#print("unadjusted upper bounds", R[8])
#print("selective mean", R[9])
#print("unadjusted mean", R[10])





