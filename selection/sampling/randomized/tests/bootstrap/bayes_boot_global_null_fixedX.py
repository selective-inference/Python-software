import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues1 import pval
from matplotlib import pyplot as plt
import regreg.api as rr


def test_lasso(s=0, n=500, p=20):

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

    #data = y.copy()
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



    def full_gradient(vec_state, obs_residuals = obs_residuals, signs=signs,
                      X=X, lam=lam, epsilon=epsilon,
                      nalpha=nalpha, active=active, inactive=inactive):

        nactive = np.sum(active); ninactive=np.sum(inactive)

        alpha = vec_state[:nalpha]
        betaE = vec_state[nalpha:(nalpha + nactive)]
        cube = vec_state[(nalpha + nactive):]

        p = X.shape[1]
        beta_full = np.zeros(p)
        beta_full[active] = betaE
        subgradient = np.zeros(p)
        subgradient[inactive] = lam*cube
        subgradient[active]  = lam*signs

        opt_vec = epsilon*beta_full+subgradient

        hessian = np.dot(X.T, X)

        #omega = -  np.dot(X.T, np.diag(obs_residuals).dot(alpha))/np.sum(alpha) + np.dot(hessian, beta_full) + opt_vec
        omega = - np.dot(X.T, np.diag(obs_residuals).dot(alpha)) + np.dot(hessian, beta_full) + opt_vec
        sign_vec = np.sign(omega)

        #sign_vec = - np.sign(gradient + opt_vec)  # sign(w), w=grad+\epsilon*beta+lambda*u

        B = hessian + epsilon * np.identity(nactive + ninactive)
        A = B[:, active]


        #mat = np.dot(X.T, np.diag(obs_residuals*np.sum(alpha)-np.multiply(obs_residuals, alpha)))
        #mat /= np.square(np.sum(alpha))
        #mat = n*mat
        mat = np.dot(X.T, np.diag(obs_residuals))
        _gradient = np.zeros(nalpha + nactive + ninactive)
        _gradient[:nalpha] = - np.ones(nalpha) + np.dot(mat.T, sign_vec)
        _gradient[nalpha:(nalpha + nactive)] = - np.dot(A.T, sign_vec)
        _gradient[(nalpha + nactive):] = - lam * sign_vec[inactive]

        return _gradient


    null, alt = pval(init_vec_state, full_gradient, full_projection, X, y, obs_residuals, nonzero, active)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(100):
        print "iteration", i
        p0, pA = test_lasso()
        if (sum(pA)>=0):
            P0.extend(p0); PA.extend(pA)

    plt.figure()
    probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=True)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.show()