import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues import pval
from matplotlib import pyplot as plt
import regreg.api as rr


def test_lasso(s=3, n=100, p=10):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    print 'sigma', sigma
    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.gaussian_Xfixed(X, y)

    random_Z = randomization.rvs(p)
    epsilon = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm_lan(p, lagrange=lam)

    #sampler1 = randomized.selective_sampler_MH_lan(loss,
    #                                           random_Z,
    #                                           epsilon,
    #                                           randomization,
    #                                          penalty)

    #loss_args = {'mean': np.zeros(n),
    #             'sigma': sigma,
    #             'linear_part':np.identity(y.shape[0]),
    #             'value': 0}

    #sampler1.setup_sampling(y, loss_args=loss_args)
    # data, opt_vars = sampler1.state

    # initial solution
    # rr.smooth_atom instead of loss?
    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, -random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)

    active = (initial_soln!=0)
    inactive = ~active
    initial_grad  = -np.dot(X.T, y-np.dot(X, initial_soln))
    betaE = initial_soln[active]
    signs = np.sign(betaE)
    subgradient = random_Z - initial_grad - epsilon*initial_soln
    cube = np.divide(subgradient[inactive],lam)
    #print betaE, cube
    #initial_grad = loss.smooth_objective(initial_soln,  mode='grad')
    #print penalty.setup_sampling(initial_grad,
    #                                     initial_soln,
    #                                     random_Z,
    #                                     epsilon)

    data0 = y.copy()
    #active = penalty.active_set

    if (np.sum(active)==0):
        print 'here'
        return [-1], [-1]

    nalpha = n
    nactive = betaE.shape[0]
    ninactive = cube.shape[0]

    alpha = np.ones(n)
    beta_bar = np.linalg.lstsq(X[:, active], y)[0]
    obs_residuals = y - np.dot(X[:, active], beta_bar)

    #obs_residuals -= np.mean(obs_residuals)
    #betaE, cube = opt_vars

    init_vec_state = np.zeros(n+nactive+ninactive)
    init_vec_state[:n] = alpha
    init_vec_state[n:(n+nactive)] = betaE
    init_vec_state[(n+nactive):] = cube


    def full_projection(vec_state, signs= signs,
                        nalpha=nalpha, nactive=nactive, ninactive = ninactive):

        alpha = vec_state[:nalpha].copy()
        betaE = vec_state[nalpha:(nalpha+nactive)]
        cube = vec_state[(nalpha+nactive):]

        #signs = penalty.signs
        projected_alpha = alpha.copy()
        projected_betaE = betaE.copy()
        projected_cube = np.zeros_like(cube)

        projected_alpha = np.clip(alpha, 0, np.inf)

        for i in range(nactive):
            if (projected_betaE[i] * signs[i] < 0):
                projected_betaE[i] = 0

        projected_cube = np.clip(cube, -1, 1)

        return np.concatenate((projected_alpha, projected_betaE, projected_cube), 0)




    null, alt = pval(init_vec_state, full_projection, X, y, obs_residuals, signs, lam, epsilon, nonzero, active)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(100):
        print "iteration", i
        p0, pA = test_lasso()
        #if (sum(pA)>=0):
        P0.extend(p0); PA.extend(pA)

    plt.figure()
    #p0 = [0.3435, 0.52, 0.7915, 0.74, 0.0585, 0.932, 0.617, 0.1245, 0.8585, 0.113, 0.651, 0.6525, 0.678, 0.9415]
    #p0.extend([0.6975, 0.004, 0.8465, 0.17, 0.8855, 0.4245, 0.0705, 0.445, 0.885, 0.626, 0.8405, 0.238, 0.387])
    probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=True)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.show()