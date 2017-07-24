from __future__ import print_function
import time

import numpy as np
from selection.reduced_optimization.initial_soln import selection, instance
from selection.reduced_optimization.ms_lasso_2stage_reduced import selection_probability_objective_ms_lasso, sel_prob_gradient_map_ms_lasso, \
    selective_map_credible_ms_lasso
from selection.reduced_optimization.tests.mixed_model import instance_mixed

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
             if u[i]<= 0.95:
                 beta_true[i] = np.random.laplace(loc=0., scale= 0.05)
             else:
                 beta_true[i] = np.random.laplace(loc=0., scale= 0.5)

         self.beta = beta_true
         #print(max(abs(self.beta)), min(abs(self.beta)))
         #print(self.beta[np.where(self.beta>3.)].shape)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X, Y, self.beta * self.sigma, self.sigma

def randomized_marginal_lasso_screening(X,
                                        y,
                                        beta,
                                        sigma):

    from selection.api import randomization

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    Z_stats = X.T.dot(y)
    randomized_Z_stats = np.true_divide(Z_stats, sigma) + random_Z

    active_1 = np.zeros(p, bool)
    active_1[np.fabs(randomized_Z_stats) > 2.33] = 1
    active_signs_1 = np.sign(randomized_Z_stats[active_1])
    nactive_1 = active_1.sum()
    threshold = 2.33 * np.ones(p)

    #print("active_1", active_1, nactive_1)

    X_step2 = X[:, active_1]
    random_Z_2 = np.random.standard_normal(nactive_1)
    sel = selection(X_step2, y, random_Z_2)
    lam, epsilon, active_2, betaE, cube, initial_soln = sel
    noise_variance = 1.
    lagrange = lam * np.ones(nactive_1)
    nactive_2 = betaE.shape[0]
    #print("active_2", active_2, nactive_2)
    active_signs_2 = np.sign(betaE)

    # getting the active indices
    active = np.zeros(p, bool)
    indices_stage2 = np.where(active_1 == 1)[0]
    active[indices_stage2[active_2]] = 1
    nactive = active.sum()
    print("the active indices after two stages of screening", active.sum())

    primal_feasible_1 = np.fabs(randomized_Z_stats[active_1])
    primal_feasible_2 = np.fabs(betaE)
    feasible_point = np.append(primal_feasible_1, primal_feasible_2)

    randomizer = randomization.isotropic_gaussian((p,), 1.)

    generative_X = X_step2[:, active_2]
    prior_variance = 1000.

    projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
    M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
    M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
    M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
    post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

    #print("observed data", post_mean)

    post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

    unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                      post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

    grad_map = sel_prob_gradient_map_ms_lasso(X,
                                              feasible_point,  # in R^{|E|_1 + |E|_2}
                                              active_1,  # the active set chosen by randomized marginal screening
                                              active_2,  # the active set chosen by randomized lasso
                                              active_signs_1,  # the set of signs of active coordinates chosen by ms
                                              active_signs_2,  # the set of signs of active coordinates chosen by lasso
                                              lagrange,  # in R^p
                                              threshold,  # in R^p
                                              generative_X,  # in R^{p}\times R^{n}
                                              noise_variance,
                                              randomizer,
                                              epsilon)

    ms = selective_map_credible_ms_lasso(y,
                                         grad_map,
                                         prior_variance)

    samples = ms.posterior_samples()

    adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

    selective_mean = np.mean(samples, axis=0)

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


if __name__ == "__main__":
    ### set parameters
    n = 1000
    p = 200
    #s = 5
    #snr = 5.

    ### GENERATE X
    np.random.seed(0)  # ensures same X

    #sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=3.)
    #sample = instance(n=n, p=p, s=s, sigma=1., rho=0)
    sample = generate_data(n, p)
    niter = 10

    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.

    for i in range(niter):

         ### GENERATE Y BASED ON SEED
         np.random.seed(i+41)  # ensures different y
         #X, y, beta, nonzero, sigma = sample.generate_response()
         X, y, beta, sigma = sample.generate_response()

         ### RUN LASSO AND TEST
         lasso = randomized_marginal_lasso_screening(X,
                                                     y,
                                                     beta,
                                                     sigma)

         if lasso is not None:
             ad_cov += lasso[0,0]
             unad_cov += lasso[1,0]
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

