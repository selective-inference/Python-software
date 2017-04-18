from __future__ import print_function

import sys
import os
import regreg.api as rr

import numpy as np
from selection.reduced_optimization.generative_model import generate_data_random

def selection(X, y, random_Z, randomization_scale=1, sigma=None, method="theoretical"):
    n, p = X.shape
    loss = rr.glm.gaussian(X,y)
    epsilon = 1. / np.sqrt(n)
    lam_frac = 1.
    if sigma is None:
        sigma = 1.
    if method == "theoretical":
        lam = 1. * sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))
    # elif method == "cross-validation":
    #     lam = tuning_parameter_glmnet(X, y)[1]
    #     print(lam)

    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p), weights = dict(zip(np.arange(p), W)), lagrange=1.)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, 0, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}


    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    active = (initial_soln != 0)
    if np.sum(active) == 0:
        return None
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE = initial_soln[active]
    subgradient = -(initial_grad+epsilon*initial_soln-randomization_scale*random_Z)
    cube = subgradient[~active]/lam
    return lam, epsilon, active, betaE, cube, initial_soln

def lasso_selection(X,
                    y,
                    beta,
                    sigma):

    from selection.api import randomization

    n,p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z)
    lam, epsilon, active, betaE, cube, initial_soln = sel

    if sel is not None:
        lagrange = lam * np.ones(p)
        active_sign = np.sign(betaE)
        nactive = active.sum()
        print("number of selected variables by Lasso", nactive)

        prior_variance = 1000.
        noise_variance = sigma**2

        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

        coverage_unad = np.zeros(nactive)
        unad_length = np.zeros(nactive)

        true_val = projection_active.T.dot(X.dot(beta))

        for l in range(nactive):
            if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                coverage_unad[l] += 1
            unad_length[l] = unadjusted_intervals[1, l] - unadjusted_intervals[0, l]

        naive_cov = coverage_unad.sum() / nactive
        unad_len = unad_length.sum() / nactive
        bayes_risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive

        return np.vstack([naive_cov, unad_len, bayes_risk_unad])

    else:
        return None


if __name__ == "__main__":
    ### set parameters
    n = 200
    p = 1000

    ### GENERATE X
    niter = 100

    unad_cov = 0.
    unad_len = 0.
    unad_risk = 0.

    for i in range(niter):

         ### GENERATE Y BASED ON SEED
         np.random.seed(i+1)  # ensures different y
         X, y, beta, sigma = generate_data_random(n=n, p=p)

         #samp = np.random.choice(range(200), 100, replace=False)
         #y_samp = y[samp]
         #X_samp = X[samp,:]

         ### RUN LASSO AND TEST
         lasso = lasso_selection(X,
                                 y,
                                 beta,
                                 sigma)

         if lasso is not None:
             unad_cov += lasso[0,0]
             unad_len += lasso[1, 0]
             unad_risk += lasso[2,0]
             print("\n")
             print("iteration completed", i)
             print("\n")

    print("unadjusted coverage, lengths and risk", unad_cov/niter, unad_len/niter, unad_risk/niter)


