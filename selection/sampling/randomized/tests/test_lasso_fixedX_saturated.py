import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues_fixedX_saturated import pval
from matplotlib import pyplot as plt
import regreg.api as rr


def selection(X, y, random_Z, sigma=1):
    n, p = X.shape
    loss = randomized.gaussian_Xfixed(X, y)
    epsilon = 1. / np.sqrt(n)
    # epsilon = 1.
    lam_frac = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    penalty = randomized.selective_l1norm_lan(p, lagrange=lam)

    # initial solution
    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0,
                                        -random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    active = (initial_soln != 0)
    if np.sum(active) == 0:
        return -1, -1, np.nan, np.nan, np.nan
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE, cube = penalty.setup_sampling(initial_grad,
                                         initial_soln,
                                         random_Z,
                                         epsilon)

    #active = penalty.active_set
    return lam, epsilon, active, betaE, cube



def test_lasso(X, y, nonzero, sigma, lam, epsilon, active, betaE, cube, random_Z,
               beta_reference,
               randomization_distribution, Langevin_steps=10000, burning=2000):

    n, p = X.shape
    step_size = 1./p

    signs = np.sign(betaE)
    inactive = ~active

    nactive = betaE.shape[0];  ninactive = cube.shape[0]
    init_vec_state = np.zeros(1+nactive+ninactive)
    init_vec_state[1:(1+nactive)] = betaE
    init_vec_state[(1+nactive):] = cube


    def full_projection(vec_state, signs=signs, nactive=nactive):

        projected_state = vec_state.copy()

        for i in range(nactive):
            if (projected_state[i+1] * signs[i] < 0):
                projected_state[i+1] = 0

        projected_state[(1+nactive):] = np.clip(projected_state[(1+nactive):], -1, 1)

        return projected_state


    null, alt, all_observed, all_variances, all_samples = pval(init_vec_state, full_projection,
                      X, y, epsilon, lam,
                      nonzero, active,
                      beta_reference,
                      Langevin_steps, burning, step_size,
                      randomization_distribution)

    return null, alt, all_observed, all_variances, all_samples


if __name__ == "__main__":

    s = 1; n = 100; p = 20
    randomization_distribution = "normal"

    P0, PA = [], []
    for i in range(10):
        print "iteration", i
        X, y, true_beta, nonzero, sigma = instance(n=n, p=p, random_signs=False, s=s, sigma=1., rho=0)
        #print "true beta", true_beta

        if randomization_distribution == "laplace":
            randomization = laplace(loc=0, scale=1.)
            random_Z = randomization.rvs(p)
        if randomization_distribution == "normal":
            random_Z = np.random.standard_normal(p)
        if randomization_distribution == "logistic":
            random_Z = np.random.logistic(loc=0, scale=1.)

        lam, epsilon, active, betaE, cube = selection(X,y,random_Z)
        if lam>0:
            beta_reference = np.zeros(np.sum(active))
            beta_reference[0]=true_beta[0]
            p0, pA, _, _, _ = test_lasso(X, y, nonzero, sigma, lam, epsilon, active, betaE, cube, random_Z,
                                         beta_reference, randomization_distribution)
            if np.sum(p0)>-1:
                P0.extend(p0); PA.extend(pA)

    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0,1], color='k', linestyle='-', linewidth=2)
    plt.suptitle("LASSO with fixed X")
    plt.show()
