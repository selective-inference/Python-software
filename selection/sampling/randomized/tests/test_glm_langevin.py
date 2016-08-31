import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.randomized import logistic_instance
from selection.sampling.randomized.losses.glm import glm
from pvalues_randomX import pval
from matplotlib import pyplot as plt
import regreg.api as rr

def test_lasso(s=5, n=200, p=20, Langevin_steps=10000, burning=2000,
               randomization_dist = "laplace", randomization_scale=1,
               covariance_estimate="nonparametric"):

    "randomization_dist: laplace or logistic"
    step_size = 1./p
    # problem setup

    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)
    print 'true_beta', beta
    nonzero = np.where(beta)[0]
    lam_frac = 1.

    if randomization_dist=="laplace":
        randomization = laplace(loc=0, scale=1.)
        random_Z = randomization.rvs(p)
    if randomization_dist=="logistic":
        random_Z = np.random.logistic(loc=0, scale=1,size=p)
    loss = glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    penalty = rr.weighted_l1norm(lam * np.ones(p), lagrange=1.)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, randomization_scale*random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}

    initial_soln = problem.solve(random_term, **solve_args)
    initial_grad = loss.gradient(initial_soln)
    print initial_soln
    active = (initial_soln != 0)
    inactive = ~active
    betaE = initial_soln[active]
    print 'betaE', betaE
    active_signs = np.sign(betaE)

    # fit restricted problem

    loss.fit_restricted(active)
    beta_unpenalized = loss._beta_unpenalized
    beta_full = np.zeros(p)
    beta_full[active] = beta_unpenalized
    null_stat = -loss.gradient(beta_full)[inactive]
    data = np.concatenate([beta_unpenalized, null_stat], axis=0)
    ndata, nactive, ninactive = p, active.sum(), p - active.sum()

    if covariance_estimate == "parametric":
        # parametric coveriance estimate
        def pi(X):
            w = np.exp(np.dot(X[:, active], beta_unpenalized))
            return w / (1 + w)

        pi_E = pi(X)
        W = np.diag(np.diag(np.outer(pi_E, 1 - pi_E)))
        Q = np.dot(X[:, active].T, np.dot(W, X[:, active]))
        Q_inv = np.linalg.inv(Q)

        mat = np.zeros((nactive+ninactive, n))
        mat[:nactive,:] = Q_inv.dot(X[:, active].T)
        mat1 = np.dot(np.dot(W,X[:, active]), np.dot(Q_inv, X[:, active].T))
        mat[nactive:,:] = X[:, inactive].T.dot(np.identity(n)-mat1)

        Sigma_full = np.dot(mat, np.dot(W, mat.T))
    else:
        # non-parametric covariance estimate
        Sigma_full = loss._cov

    Sigma_full_inv = np.linalg.inv(Sigma_full)

    # initialize the sampler

    mle_slice = slice(0, nactive)
    null_slice = slice(nactive, p)
    beta_slice = slice(p, p + nactive)
    subgrad_slice = slice(p + nactive, 2*p)

    init_state = np.zeros(2*p)
    init_state[mle_slice] = beta_unpenalized
    init_state[null_slice] = null_stat
    init_state[beta_slice] = betaE
    init_state[subgrad_slice] = -initial_grad[inactive] / (penalty.weights[inactive] * penalty.lagrange)

    # define the projection map for langevin

    def full_projection(state):

        new_state = state.copy()
        new_beta = new_state[beta_slice]
        new_subgrad = new_state[subgrad_slice]

        new_beta[:] = new_beta * (new_beta * active_signs >= 0)
        new_subgrad[:] = np.clip(new_subgrad, -1, 1)

        return new_state

    # form affine map in argument of `g` (randomization density) for selective sampler

    linear_term = np.zeros((p, 2*p))

    # hessian part
    H = loss._restricted_hessian
    linear_term[:,mle_slice] = -H
    linear_term[:,beta_slice] = H 

    # null part
    linear_term[nactive:][:,null_slice] = -np.identity(ninactive)

    # quadratic part
    linear_term[:nactive][:,beta_slice] = linear_term[:nactive][:,beta_slice] + epsilon * np.identity(nactive)

    # subgrad part
    linear_term[nactive:][:,subgrad_slice] = linear_term[nactive:][:,subgrad_slice] + np.diag(penalty.lagrange * penalty.weights[inactive])

    affine_term = np.zeros(p)
    affine_term[:nactive] = penalty.lagrange * penalty.weights[active] * active_signs

    # define the gradient

    def full_gradient(state, loss=loss, penalty = penalty, Sigma_full_inv=Sigma_full_inv,
                      lam=lam, epsilon=epsilon, ndata=ndata, active=active, inactive=inactive):

        # affine reconstruction map
        omega = linear_term.dot(state) + affine_term

        if randomization_dist == "laplace":
            randomization_derivative = np.sign(omega)/randomization_scale

        if randomization_dist == "logistic":
            omega_scaled = omega/randomization_scale
            randomization_derivative = -(np.exp(-omega_scaled)-1)/(np.exp(-omega_scaled)+1)
            randomization_derivative /= randomization_scale

        _gradient = -linear_term.T.dot(randomization_derivative)

        # now add in the Gaussian derivative

        _gradient[:p] -= np.dot(Sigma_full_inv, data)

        return _gradient


    null, alt = pval(init_state, full_gradient, full_projection,
                      Sigma_full[:nactive, :nactive], data, nonzero, active,
                     Langevin_steps, burning, step_size)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []

    plt.figure()
    plt.ion()

    for i in range(20):
        print "iteration", i
        p0, pA = test_lasso()
        P0.extend(p0); PA.extend(pA)
        plt.clf()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
        plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
        plt.pause(0.01)

    while True:
        plt.pause(0.05)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0,1], color='k', linestyle='-', linewidth=2)
    plt.show()
