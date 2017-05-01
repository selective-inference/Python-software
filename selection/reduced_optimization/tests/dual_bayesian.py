from __future__ import print_function
import sys
import os
import time

import numpy as np
from selection.bayesian.initial_soln import selection, instance

from selection.reduced_optimization.dual_lasso import selection_probability_lasso_dual, sel_prob_gradient_map_lasso, selective_inf_lasso

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

def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma):

    from selection.api import randomization

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
        bayes_risk_ad = np.power(selective_mean - true_val, 2.).sum() / nactive
        bayes_risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive

        return np.vstack([sel_cov, naive_cov, ad_len, unad_len, bayes_risk_ad, bayes_risk_unad])

    else:
        return np.vstack([0., 0., 0., 0., 0., 0.])

# if __name__ == "__main__":
# # read from command line
#     seedn=int(sys.argv[1])
#     outdir=sys.argv[2]
#
#     outfile = os.path.join(outdir, "list_result_" + str(seedn) + ".txt")
#
# ### set parameters
#     n = 1000
#     p = 200
#     s = 0
#     snr = 5.
#
# ### GENERATE X
#     np.random.seed(0)  # ensures same X
#
#     sample = generate_data(n, p)
#
# ### GENERATE Y BASED ON SEED
#     np.random.seed(seedn) # ensures different y
#     X, y, beta, sigma = sample.generate_response()
#
#     lasso = randomized_lasso_trial(X,
#                                    y,
#                                    beta,
#                                    sigma)
#
#     np.savetxt(outfile, lasso)

if __name__ == "__main__":
    ### set parameters
    n = 1000
    p = 200
    s = 0
    snr = 5.

    ### GENERATE X
    np.random.seed(0)  # ensures same X

    sample = generate_data(n, p)

    niter = 20

    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.

    for i in range(niter):

         ### GENERATE Y BASED ON SEED
         np.random.seed(i+30)  # ensures different y
         X, y, beta, sigma = sample.generate_response()

         ### RUN LASSO AND TEST
         lasso = randomized_lasso_trial(X,
                                        y,
                                        beta,
                                        sigma)

         if lasso is not None:
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