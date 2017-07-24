from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
from selection.reduced_optimization.initial_soln import selection
from selection.tests.instance import logistic_instance, gaussian_instance

from selection.reduced_optimization.par_carved_reduced import selection_probability_carved, sel_inf_carved

#from selection.reduced_optimization.estimator import M_estimator_approx_carved
from selection.randomized.M_estimator import M_estimator, M_estimator_split
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov

import sys
import os

def generate_data_random(n, p, sigma=1., rho=0., scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None, :] * np.sqrt(n))

    beta_true = np.zeros(p)
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.9:
            beta_true[i] = np.random.laplace(loc=0., scale=0.1)
        else:
            beta_true[i] = np.random.laplace(loc=0., scale=1.)

    beta = beta_true

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta * sigma, sigma

class M_estimator_approx_carved(M_estimator_split):

    def __init__(self, loss, epsilon, subsample_size, penalty, estimation):

        M_estimator_split.__init__(self,loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10})
        self.estimation = estimation

    def solve_approx(self):

        self.solve()

        self.nactive = self._overall.sum()
        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        self.target_observed = self.observed_score_state[:self.nactive]

        self.feasible_point = np.concatenate([self.observed_score_state, np.fabs(self.observed_opt_state[:self.nactive]),
                                              self.observed_opt_state[self.nactive:]], axis = 0)

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)

        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)

        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)

        #print("True or false", np.all(lagrange[0]-np.fabs(self.feasible_point[p+self.nactive:]))>0)
        #print("True or false", np.all(self.feasible_point[p:][:self.nactive]) > 0)

        self.inactive_lagrange = lagrange[~self._overall]

        self.bootstrap_score, self.randomization_cov = self.setup_sampler()

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))
            inv_X_active = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
            projection_X_active = X[:,self._overall].dot(np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))).dot(X[:,self._overall].T)
            score_cov[:self.nactive, :self.nactive] = inv_X_active
            score_cov[self.nactive:, self.nactive:] = X[:,~self._overall].T.dot(np.identity(n)- projection_X_active).dot(X[:,~self._overall])

        elif self.estimation == 'bootstrap':
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), self.bootstrap_score)

        self.score_cov = score_cov
        self.score_cov_inv = np.linalg.inv(self.score_cov)

def carved_lasso_trial(X,
                       y,
                       beta,
                       sigma,
                       lam,
                       estimation='parametric'):
    n, p = X.shape

    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

    total_size = loss.saturated_loss.shape[0]
    subsample_size = int(0.8 * total_size)

    M_est = M_estimator_approx_carved(loss, epsilon, subsample_size, penalty, estimation)

    M_est.solve_approx()
    active = M_est._overall
    nactive = M_est.nactive

    if nactive >= 1:
        prior_variance = 1000.
        noise_variance = sigma ** 2
        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

        print("observed data", post_mean)

        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])
        grad_lasso = sel_inf_carved(M_est, prior_variance)
        samples = grad_lasso.posterior_samples()
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

    else:
        return np.vstack([0.,0.,0.,0.,0.,0.])


if __name__ == "__main__":

    # # read from command line
    # seedn = int(sys.argv[1])
    # outdir = sys.argv[2]
    #
    # outfile = os.path.join(outdir, "list_result_" + str(seedn) + ".txt")

    ### set parameters
    n = 1000
    p = 100
    s = 0
    snr = 0.

    niter = 50
    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.

    for i in range(niter):
        np.random.seed(i)
        X, y, beta, sigma = generate_data_random(n=n, p=p)
        lam = 0.8 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        lasso = carved_lasso_trial(X,
                                   y,
                                   beta,
                                   sigma,
                                   lam)

        ad_cov += lasso[0, 0]
        unad_cov += lasso[1, 0]
        ad_len += lasso[2, 0]
        unad_len += lasso[3, 0]
        ad_risk += lasso[4, 0]
        unad_risk += lasso[5, 0]

        print("\n")
        print("iteration completed", i)
        print("adjusted and unadjusted coverage", ad_cov, unad_cov)
        print("adjusted and unadjusted lengths", ad_len, unad_len)
        print("adjusted and unadjusted risks", ad_risk, unad_risk)

    print("adjusted and unadjusted coverage", ad_cov, unad_cov)
    print("adjusted and unadjusted lengths", ad_len, unad_len)
    print("adjusted and unadjusted risks", ad_risk, unad_risk)
