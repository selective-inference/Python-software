from __future__ import division


import numpy as np
from scipy.stats import laplace, probplot, uniform
import scipy.stats as stats

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from matplotlib import pyplot as plt
from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin

import regreg.api as rr

def projection_cone(p, max_idx, max_sign):
    """

    Create a callable that projects onto one of two cones,
    determined by which coordinate achieves the max in one
    step of forward stepwise.

    Parameters
    ----------

    p : int
        Dimension.

    max_idx : int
        Index achieving the max.

    max_sign : [-1,1]
        Sign of achieved max.


    Returns
    -------

    projection : callable
        A function to compute projection onto appropriate cone.
        Takes one argument of shape (p,).

    """

    if max_sign > 0:
        P = rr.linf_epigraph(p-1)
    else:
        P = rr.l1_epigraph_polar(p-1)

    def _projection(state):
        permuted_state = np.zeros_like(state)
        permuted_state[-1] = state[max_idx]
        permuted_state[:max_idx] = state[:max_idx]
        permuted_state[max_idx:-1] = state[(max_idx+1):]

        projected_state = P.cone_prox(permuted_state)

        new_state = np.zeros_like(state)
        new_state[max_idx] = projected_state[-1]
        new_state[:max_idx] = projected_state[:max_idx]
        new_state[(max_idx+1):] = projected_state[max_idx:-1]

        return new_state

    return _projection

def test_fstep(s=0, n=50, p=10, weights = "gumbel", randomization_dist ="logistic",
               Langevin_steps = 10000, burning=1000):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    epsilon = 0.
    if randomization_dist == "laplace":
        randomization = laplace(loc=0, scale=1.)
        random_Z = randomization.rvs(p)
    if randomization_dist=="logistic":
        random_Z = np.random.logistic(loc=0, scale=1, size=p)

    T = np.dot(X.T,y)
    T_random = T + random_Z
    T_abs = np.abs(T_random)
    j_star = np.argmax(T_abs)
    s_star = np.sign(T_random[j_star])

    # this is the subgradient part of the projection
    projection = projection_cone(p, j_star, s_star)


    def full_projection(state, n=n, p=p):
        """
        State is (y, u) -- first n coordinates are y, last p are u.
        """
        new_state = np.empty(state.shape, np.float)
        new_state[:n] = state[:n]
        new_state[n:] = projection(state[n:])
        return new_state


    obs = np.max(np.abs(T))
    eta_star = np.zeros(p)
    eta_star[j_star] = s_star


    def full_gradient(state, n=n, p=p, X=X, y=y):
        #data = state[:n]

        alpha = state[:n]
        subgrad = state[n:]

        mat = np.dot(X.T, np.diag(y))
        omega = - mat.dot(alpha) + subgrad

        if randomization_dist == "laplace":
            randomization_derivative = np.sign(omega)
        if randomization_dist == "logistic":
            randomization_derivative = -(np.exp(-omega) - 1) / (np.exp(-omega) + 1)
        if randomization_dist == "normal":
            randomization_derivative = omega

        grad = np.empty(state.shape, np.float)
        #grad[:n] = - (data - X.dot(randomization_derivative))
        grad[:n] = np.dot(mat.T,randomization_derivative)

        if weights == "normal":
            grad[:n] -= alpha
        if (weights == "gumbel"):
            gumbel_beta = np.sqrt(6) / (1.14 * np.pi)
            euler = 0.57721
            gumbel_mu = -gumbel_beta * euler
            gumbel_sigma = 1. / 1.14
            grad[:n] -= (1. - np.exp(-(alpha * gumbel_sigma - gumbel_mu) / gumbel_beta)) * gumbel_sigma / gumbel_beta

        grad[n:] = - randomization_derivative

        return grad



    state = np.zeros(n+p)
    #state[:n] = y
    state[:n] = np.zeros(n)
    state[n:] = T_random

    sampler = projected_langevin(state,
                                 full_gradient,
                                 full_projection,
                                 1./p)
    samples = []

    for i in range(Langevin_steps):
        sampler.next()
        if (i>burning):
            samples.append(sampler.state.copy())

    samples = np.array(samples)
    Z = samples[:,:n]
    print Z.shape

    mat = np.dot(X.T,np.diag(y))

    #pop = [np.linalg.norm(np.dot(mat, Z[i,:].T)) for i in range(Z.shape[0])]
    #obs = np.linalg.norm(np.dot(X.T,y))
    pop = np.abs(np.dot(mat, Z.T)).max(0)
    fam = discrete_family(pop, np.ones_like(pop))
    pval = fam.cdf(0, obs)
    pval = 2 * min(pval, 1 - pval)

    #stop

    print 'pvalue:', pval
    return pval


def plot(data, cdf=stats.uniform.cdf):
    plt.figure()
    plt.plot(np.cumsum(1./len(data) * np.ones(len(data))), cdf(np.sort(data)),'ro', markersize=20)
    plt.plot((0,1),(0,1),'r-')

    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.xlabel('empirical percentiles')
    plt.ylabel('fit percentiles')
    plt.show()

if __name__ == "__main__":

    P0 = []
    plt.figure()
    plt.ion()
    for i in range(100):
        print "iteration", i
        pval = test_fstep()
        P0.append(pval)
        plt.clf()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot(markersize = 50)
        probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
        plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
        plt.pause(0.01)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)

    while True:
        plt.pause(0.05)
    plt.show()

    #plot(P0)

