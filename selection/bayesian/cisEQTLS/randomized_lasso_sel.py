from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import regreg.api as rr

#arguments of function are X with normalized columns, response y, sigma_hat and randomization
def selection(X, y, random_Z, randomization_scale=None, sigma=None, lam=None):
    n, p = X.shape
    loss = rr.glm.gaussian(X,y)
    lam_frac = 1.
    if sigma is None:
        sigma = 1.
    if lam is None:
        lam = 0.8* sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))
    if randomization_scale is None:
        randomization_scale = sigma

    epsilon = 1. / np.sqrt(n)
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p), weights = dict(zip(np.arange(p), W)), lagrange=1.)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, -randomization_scale * random_Z, 0)
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





