from __future__ import print_function
import time
import sys
import os

import numpy as np
from selection.bayesian.initial_soln import selection, instance
from selection.reduced_optimization.forward_stepwise_reduced import neg_log_cube_probability_fs, \
    selection_probability_objective_fs, sel_prob_gradient_map_fs, selective_map_credible_fs

class generate_data():

    def __init__(self, n, p, sigma=1., rho=0., scale =True, center=True):
         (self.n, self.p, self.sigma, self.rho) = (n, p, sigma, rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
                   np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         beta_true = np.zeros(p)
         u = np.random.uniform(0.,1.,p)
         for i in range(p):
             if u[i]<= 0.9:
                 beta_true[i] = np.random.laplace(loc=0., scale=0.1)
             else:
                 beta_true[i] = np.random.laplace(loc=0., scale=1.)

         self.beta = beta_true

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X, Y, self.beta * self.sigma, self.sigma

def randomized_forward_step(X,
                            y,
                            beta,
                            sigma):
    from selection.api import randomization

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    Z_stats = X.T.dot(y)
    random_obs = X.T.dot(y) + random_Z

    active_index = np.argmax(np.fabs(random_obs))
    active = np.zeros(p, bool)
    active[active_index] = 1
    active_sign = np.sign(random_obs[active_index])
    print("observed statistic", random_obs[active_index], Z_stats[active_index])
    print("first step--chosen index and sign", active_index, active_sign)

    feasible_point = np.fabs(random_obs[active_index])

    noise_variance = sigma ** 2

    randomizer = randomization.isotropic_gaussian((p,), 1.)

    generative_X = X[:, active]
    prior_variance = 1000.

    grad_map = sel_prob_gradient_map_fs(X,
                                        feasible_point,
                                        active,
                                        active_sign,
                                        generative_X,
                                        noise_variance,
                                        randomizer)

    inf = selective_map_credible_fs(y, grad_map, prior_variance)

    samples = inf.posterior_samples()

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

    coverage_ad = np.zeros(1)
    coverage_unad = np.zeros(1)
    ad_length = np.zeros(1)
    unad_length = np.zeros(1)

    true_val = projection_active.T.dot(X.dot(beta))


    if (adjusted_intervals[0, 0] <= true_val[0]) and (true_val[0] <= adjusted_intervals[1, 0]):
        coverage_ad[0] += 1

    ad_length[0] = adjusted_intervals[1, 0] - adjusted_intervals[0, 0]
    if (unadjusted_intervals[0, 0] <= true_val[0]) and (true_val[0] <= unadjusted_intervals[1, 0]):
        coverage_unad[0] += 1

    unad_length[0] = unadjusted_intervals[1, 0] - unadjusted_intervals[0, 0]

    sel_cov = coverage_ad.sum() / 1.
    naive_cov = coverage_unad.sum() / 1.
    ad_len = ad_length.sum() / 1.
    unad_len = unad_length.sum() / 1.
    bayes_risk_ad = np.power(selective_mean - true_val, 2.).sum() / 1.
    bayes_risk_unad = np.power(post_mean - true_val, 2.).sum() / 1.

    return np.vstack([sel_cov, naive_cov, ad_len, unad_len, bayes_risk_ad, bayes_risk_unad])

def test_FS():

    n = 200
    p = 1000
    s = 0
    snr = 5.

    niter = 50
    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.

    ### GENERATE X
    np.random.seed(0)  # ensures same X

    sample = generate_data(n, p)

    ### GENERATE Y BASED ON SEED
    for i in range(niter):
        np.random.seed(i) # ensures different y
        X, y, beta, sigma = sample.generate_response()
        lasso = randomized_forward_step(X,
                                        y,
                                        beta,
                                        sigma)

        ad_cov += lasso[0, 0]
        unad_cov += lasso[1, 0]
        ad_len += lasso[2, 0]
        unad_len += lasso[3, 0]
        ad_risk += lasso[4, 0]
        unad_risk += lasso[5, 0]

        print("\n")
        print("iteration completed", i)
        print("\n")
        print("adjusted and unadjusted coverage", ad_cov, unad_cov)
        print("adjusted and unadjusted lengths", ad_len, unad_len)
        print("adjusted and unadjusted risks", ad_risk, unad_risk)

    print("adjusted and unadjusted coverage", ad_cov, unad_cov)
    print("adjusted and unadjusted lengths", ad_len, unad_len)
    print("adjusted and unadjusted risks", ad_risk, unad_risk)

    #np.savetxt(outfile, lasso)
