from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.inference_rr import sel_prob_gradient_map, selective_map_credible

def test_inf_regreg(n, p, s, snr):

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, true_beta, active, active.sum())
    noise_variance = 1.
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    active_set = [i for i in range(p) if active[i]]
    true_val = true_beta[active]
    true_support = np.asarray([i for i in range(p) if i < s])

    primal_feasible = np.fabs(betaE)
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
    lagrange = lam * np.ones(p)
    generative_X = X_1[:, active]
    prior_variance = 1000.

    Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
    post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
    post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
    unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])

    if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:

        inf_rr = selective_map_credible(y,
                                        X_1,
                                        primal_feasible,
                                        dual_feasible,
                                        active,
                                        active_signs,
                                        lagrange,
                                        generative_X,
                                        noise_variance,
                                        prior_variance,
                                        randomization.isotropic_gaussian((p,), tau),
                                        epsilon)

        samples = inf_rr.posterior_samples()

        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        print(active)
        print("adjusted intervals", adjusted_intervals)
        print("usual intervals", unadjusted_intervals)
        return unadjusted_intervals, adjusted_intervals, active_set, true_val, nactive

    else:
        return 0

def compute_coverage(n= 100, p = 30 , s = 0, snr = 5):

    niter = 50
    coverage_ad = np.zeros(p)
    coverage_unad = np.zeros(p)
    nsel = np.zeros(p)
    nerr = 0
    for iter in range(niter):
        print("\n")
        print("iteration", iter)
        try:
            test_cred = test_inf_regreg(n= 100, p = 30 , s = 0, snr = 5)
            cred_active_ad = test_cred[1]
            cred_active_unad = test_cred[0]
            print("cred- adjusted", cred_active_ad)
            active_set = np.asarray(test_cred[2])
            true_val = test_cred[3]
            nactive = test_cred[4]
            for l in range(nactive):
                nsel[active_set[l]] += 1
                if (cred_active_ad[0,l]<= true_val[l]) and (true_val[l]<= cred_active_ad[1,l]):
                    coverage_ad[active_set[l]] += 1
                if (cred_active_unad[0,l]<= true_val[l]) and (true_val[l]<= cred_active_unad[1,l]):
                    coverage_unad[active_set[l]] += 1
            print('coverage adjusted so far',np.true_divide(coverage_ad, nsel))
            print('coverage unadjusted so far', np.true_divide(coverage_unad, nsel))

        except ValueError:
            nerr +=1
            print('ignore iteration raising ValueError')
            continue

    coverage_prop_ad = np.true_divide(coverage_ad, nsel)
    coverage_prop_ad[coverage_prop_ad == np.inf] = 0
    coverage_prop_ad = np.nan_to_num(coverage_prop_ad)

    coverage_prop_unad = np.true_divide(coverage_unad, nsel)
    coverage_prop_unad[coverage_prop_unad == np.inf] = 0
    coverage_prop_unad = np.nan_to_num(coverage_prop_unad)

    return coverage_prop_ad.sum()/p, coverage_prop_unad.sum()/p, nsel, nerr

compute_coverage()
