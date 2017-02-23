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


def one_trial(n=150, p= 100, s= 10, snr = 5., seed_n = 19, method="theoretical"):

    random.seed(seed_n)

    sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

    X, y, true_beta, nonzero, noise_variance = sample.generate_response()

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z)
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

        Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
        post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
        post_var = prior_variance * np.identity(nactive) - ((prior_variance ** 2) * (generative_X.T.dot(Q).dot(generative_X)))
        unadjusted_intervals = np.vstack([post_mean - 1.65 * (post_var.diagonal()), post_mean + 1.65 * (post_var.diagonal())])

        coverage_ad = np.zeros(p)
        coverage_unad = np.zeros(p)
        nerr = 0.
        true_val = true_beta[active]
        active_set = [i for i in range(p) if active[i]]
        active_ind = np.zeros(p)
        active_ind[active_set] = 1

        list_results = []
        list_results.append(true_beta)
        list_results.append(active_ind)

        if nactive > 1:
            try:
                for l in range(nactive):
                    if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                        coverage_ad[active_set[l]] += 1
                    if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                        coverage_unad[active_set[l]] += 1

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
            p_BH = BH_q(p_value, 0.05)

            D_BH = np.zeros(p)
            fD_BH = np.zeros(p)

            if p_BH is not None:

                indices_sig = active_set[p_BH[1]]
                indices_nsig = np.setdiff1d(active_set, indices_sig)

                sig_total = indices_sig.shape[0]
                D_BH[indices_sig] = 1
                fD_BH[indices_nsig] =1

            list_results.append(D_BH)
            list_results.append(fD_BH)

            return list.results



R = one_trial()
print("true parameter",R[0])
print("active indices",R[1])
print("indices covered by adjusted",R[2])
print("indices covered by unadjusted",R[3])
print("indices declared significant after BH",R[4])
print("indices declared insignificant after BH", R[5])



