from __future__ import print_function
import time

import numpy as np
from selection.tests.instance import gaussian_instance
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.bayesian.cisEQTLS.inference_per_gene import selection_probability_variants, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
from selection.bayesian.cisEQTLS.initial_sol_wocv import selection, instance


def test_coverage():
    n = 350
    p = 5000
    s = 10
    snr = 5.

    X, y, true_beta, nonzero, noise_variance = sample.generate_response()

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z)
    lam, epsilon, active, betaE, cube, initial_soln = sel

    if sel is not None:
        lagrange = lam * np.ones(p)
        active_sign = np.sign(betaE)
        nactive = active.sum()
        print("number of selected variables by Lasso", nactive)

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

        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)
        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)
        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

        coverage_ad = np.zeros(p)
        coverage_unad = np.zeros(p)
        nerr = 0.
        #true_val = true_beta[active]
        true_val = projection_active.T.dot(X.dot(true_beta))

        active_set = [i for i in range(p) if active[i]]

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

            no_BH_results = coverage_ad.sum() / nactive

            ngrid = 1000
            quantiles = np.zeros((ngrid, nactive))
            for i in range(ngrid):
                quantiles[i, :] = np.percentile(samples, (i * 100.) / ngrid, axis=0)

            index_grid = np.argmin(np.abs(quantiles - np.zeros((ngrid, nactive))), axis=0)
            p_value = 2 * np.minimum(np.true_divide(index_grid, ngrid), 1. - np.true_divide(index_grid, ngrid))
            p_BH = BH_q(p_value, 0.10)

            # print("adjusted BH intervals", adjusted_intervals[:, p_BH[1]])
            D_BH = 0.
            fD_BH = 0.

            if p_BH is not None:

                indices_sig = p_BH[1]
                indices_nsig = np.setdiff1d(np.arange(nactive), indices_sig)

                sig_total = indices_sig.shape[0]
                for l in range(sig_total):
                    if true_val[indices_sig[l]] > 0:
                        D_BH += 1
                    else:
                        fD_BH += 1

                BH_D = [D_BH, fD_BH]

            else:
                BH_D = [0., 0.]

            return no_BH_results, BH_D

        else:
            return None



cov_ad = 0.
BH_D = 0.
fD = 0.
tD = 0.
n = 350
p = 5000
s = 10
snr = 5.

sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
niter = 15
for i in range(niter):

    cov = test_coverage()
    if cov is not None:
        cov_ad += cov[0]
        BH_D = cov[1]
        fD += BH_D[1] / max(float(BH_D[1] + BH_D[0]), 1.)
        tD += BH_D[0] / 5.
        tD += 0.

        print('coverage adjusted so far', cov_ad)
        print('fDR and power', fD, tD)
        print("\n")
        print("iteration completed", i)


