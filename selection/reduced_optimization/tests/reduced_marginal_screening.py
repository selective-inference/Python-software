from __future__ import print_function
import time

import numpy as np
from selection.reduced_optimization.initial_soln import selection, instance
from selection.reduced_optimization.marginal_screening_reduced import selection_probability_ms, sel_prob_gradient_map_ms, selective_inf_ms

def randomized_marginal_screening(X,
                                  y,
                                  beta,
                                  sigma):

    from selection.api import randomization

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    Z_stats = X.T.dot(y)
    randomized_Z_stats = np.true_divide(Z_stats, sigma) + random_Z

    active = np.zeros(p, bool)
    active[np.fabs(randomized_Z_stats) > 2.33] = 1
    active_signs = np.sign(randomized_Z_stats[active])
    nactive = active.sum()
    threshold = 2.33 * np.ones(p)

    if nactive >= 1:

        feasible_point = np.fabs(randomized_Z_stats[active])

        noise_variance = sigma ** 2

        randomizer = randomization.isotropic_gaussian((p,), 1.)

        generative_X = X[:, active]
        prior_variance = 1000.

        grad_map = sel_prob_gradient_map_ms(X,
                                            feasible_point,
                                            active,
                                            active_signs,
                                            threshold,
                                            generative_X,
                                            noise_variance,
                                            randomizer)

        inf = selective_inf_ms(y, grad_map, prior_variance)

        samples = inf.posterior_samples()

        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

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
        nerr = 0.

        true_val = projection_active.T.dot(X.dot(beta))

        active_set = [i for i in range(p) if active[i]]


        for l in range(nactive):
            if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                coverage_ad[l] += 1
            if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                coverage_unad[l] += 1


        sel_cov = coverage_ad.sum() / nactive
        naive_cov = coverage_unad.sum() / nactive

        return sel_cov, naive_cov

    else:
        return None


if __name__ == "__main__":
    ### set parameters
    n = 200
    p = 1000
    s = 0
    snr = 5.

    ### GENERATE X
    np.random.seed(0)  # ensures same X

    sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)

    niter = 10

    ad_cov = 0.
    unad_cov = 0.

    for i in range(niter):

         ### GENERATE Y BASED ON SEED
         np.random.seed(i+1)  # ensures different y
         X, y, beta, nonzero, sigma = sample.generate_response()

         ### RUN LASSO AND TEST
         lasso = randomized_marginal_screening(X,
                                               y,
                                               beta,
                                               sigma)

         if lasso is not None:
             ad_cov += lasso[0]
             unad_cov += lasso[1]
             print("\n")
             print("iteration completed", i)
             print("\n")
             print("adjusted and unadjusted coverage", ad_cov, unad_cov)


    print("adjusted and unadjusted coverage",ad_cov, unad_cov)
