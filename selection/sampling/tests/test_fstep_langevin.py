from __future__ import print_function
import numpy as np
from scipy.stats import laplace, uniform

from selection.tests.instance import gaussian_instance
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

def projection_cone_nosign(p, max_idx):
    """

    Does not condition on sign!

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

    P_plus = rr.linf_epigraph(p-1)
    P_minus = rr.l1_epigraph_polar(p-1)

    def _projection(state):
        permuted_state = np.zeros_like(state)
        permuted_state[-1] = state[max_idx] 
        permuted_state[:max_idx] = state[:max_idx]
        permuted_state[max_idx:-1] = state[(max_idx+1):]

        projected_state_plus = P_plus.cone_prox(permuted_state)
        projected_state_minus = P_minus.cone_prox(permuted_state)

        D_plus = np.linalg.norm(permuted_state - projected_state_plus)
        D_minus = np.linalg.norm(permuted_state - projected_state_minus)

        if D_plus < D_minus:
            projected_state = projected_state_plus
        else:
            projected_state = projected_state_minus

        new_state = np.zeros_like(state)
        new_state[max_idx] = projected_state[-1]
        new_state[:max_idx] = projected_state[:max_idx]
        new_state[(max_idx+1):] = projected_state[max_idx:-1]

        return new_state

    return _projection

def test_fstep(s=0, n=100, p=10, Langevin_steps=10000, burning=2000, condition_on_sign=True):

    X, y, _, nonzero, sigma = gaussian_instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    epsilon = 0.
    randomization = laplace(loc=0, scale=1.)

    random_Z = randomization.rvs(p)
    T = np.dot(X.T,y) 
    T_random = T + random_Z
    T_abs = np.abs(T_random)
    j_star = np.argmax(T_abs)
    s_star = np.sign(T_random[j_star])

    # this is the subgradient part of the projection

    if condition_on_sign:
        projection = projection_cone(p, j_star, s_star)
    else:
        projection = projection_cone_nosign(p, j_star)

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

    def full_gradient(state, n=n, p=p, X=X):
        data = state[:n]
        subgrad = state[n:]
        sign_vec = np.sign(-X.T.dot(data) + subgrad)

        grad = np.empty(state.shape, np.float)
        grad[n:] = - sign_vec

        grad[:n] = - (data - X.dot(sign_vec))
        return grad



    state = np.zeros(n+p)
    state[:n] = y
    state[n:] = T_random

    sampler = projected_langevin(state,
                                 full_gradient,
                                 full_projection,
                                 1./p)
    samples = []

    for i in range(Langevin_steps):
        if i>burning:
            sampler.next()
            samples.append(sampler.state.copy())

    samples = np.array(samples)
    Z = samples[:,:n]

    pop = np.abs(X.T.dot(Z.T)).max(0)
    fam = discrete_family(pop, np.ones_like(pop))
    pval = fam.cdf(0, obs)
    pval = 2 * min(pval, 1 - pval)

    #stop

    print('pvalue:', pval)
    return pval


def main():

    P0 = []
    for i in range(100):
        print("iteration", i)
        #print form_Ab(1,4)
        pval = test_fstep(condition_on_sign=True)

        P0.append(pval)

    print("done! mean: ", np.mean(P0), "std: ", np.std(P0))

    import statsmodels.api as sm
    U = np.linspace(0, 1, 101)
    plt.clf()
    plt.plot(U, sm.distributions.ECDF(P0)(U))
    plt.show()


