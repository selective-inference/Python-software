from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.inference_fs import sel_prob_gradient_map_fs, selective_map_credible_fs

def test_inf_fs():
    n = 100
    p = 50
    s = 0
    snr = 5

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    #print(true_beta, nonzero, noise_variance)
    random_Z = np.random.standard_normal(p)
    random_obs = X_1.T.dot(y) + random_Z
    active_index = np.argmax(random_obs)
    active = np.zeros(p, bool)


    active[active_index] = 1
    active_sign = np.sign(random_obs[active_index])
    nactive = 1
    #X_active = X_1[active_index]
    #obs = np.linalg.inv(X_active.T.dot(X_active)).dot(X_active.T.dot(y))

    #print("observed data point", obs)

    active_set = [i for i in range(p) if active[i]]
    true_support = np.asarray([i for i in range(p) if i < s])

    primal_feasible = np.fabs(random_obs[active_index])
    tau = 1  # randomization_variance

    parameter = np.random.standard_normal(nactive)
    mean = X_1[:, active].dot(parameter)

    generative_X = X_1[:, active]
    prior_variance = 1000.

    true_vec = true_beta[active]

    Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
    post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
    post_var = prior_variance * np.identity(nactive) - (
    (prior_variance ** 2) * (generative_X.T.dot(Q).dot(generative_X)))
    unadjusted_intervals = np.hstack([post_mean - 1.65 * (post_var.diagonal()), post_mean + 1.65 * (post_var.diagonal())])

    if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:

        inf_rr = selective_map_credible_fs(y,
                                           X_1,
                                           primal_feasible,
                                           active,
                                           active_sign,
                                           generative_X,
                                           noise_variance,
                                           prior_variance,
                                           randomization.isotropic_gaussian((p,), tau))


        samples = inf_rr.posterior_samples()

        adjusted_intervals = np.hstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        return np.hstack([adjusted_intervals, unadjusted_intervals])

    else:
        return 0

#print(test_inf_fs())


coverage_ad = 0.
coverage_unad = 0.
niter = 100
nerr = 0
for i in range(niter):
    print("\n")
    print("iteration", i)
    try:

        cred = test_inf_fs()
        if cred != 0:
            print("intervals", cred)
            if (cred[0]<=0) and (cred[1]>=0):
                coverage_ad += 1
            if (cred[2]<=0) and (cred[3]>=0):
                coverage_unad += 1

    except ValueError:
        nerr += 1
        print('ignore iteration raising ValueError')
        continue

    print("adjusted proportion", coverage_ad/(i+1-nerr))
    print("unadjusted proportion", coverage_unad/(i + 1 - nerr))

print(coverage_ad/(niter-nerr))