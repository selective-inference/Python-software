import numpy as np



from scipy.stats import laplace, probplot, uniform

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


def test_2fstep(s=0, n=100, p=10):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    epsilon = 0.
    randomization = laplace(loc=0, scale=1.)
    k = 5

    j_seq = np.empty(k, dtype=int)
    s_seq = np.empty(k)

    left = np.ones(p, dtype=bool)
    obs = np.zeros((p,k))

    state = np.zeros(n + np.sum([i for i in range(p-k+1,p+1)]))

    state[:n] = y.copy()

    mat = [np.array((n, ncol)) for ncol in range(p,p-k,-1)]

    curr = n
    for i in range(k):
        X_left = X[:,left]
        X_used = X[:, ~left]
        if (np.sum(left)<p):
            P_perp = np.identity(X_used.shape[0]) - X_used.dot(np.linalg.pinv(X_used))
            y_mod = P_perp.dot(y)
            mat[i] = P_perp.dot(X_left)
        else:
            y_mod = y
            mat[i] = X

        T = np.dot(X_left.T,y_mod)
        obs = np.max(np.abs(T))

        random_Z = randomization.rvs(T.shape[0])

        T_random = T + random_Z
        state[curr:(curr+p-i)] = T_random
        curr = curr + p-i

        T_abs = np.abs(T_random)
        j_seq[i] = np.argmax(T_abs)
        s_seq[i] = np.sign(T_random[j_seq[i]])

        #print j_seq[i]

        def find_index(v, idx):
            _sumF = 0
            _sumT = 0
            for i in range(v.shape[0]):
                if (v[i] == False):
                    _sumF = _sumF + 1
                else:
                    _sumT = _sumT + 1
                if _sumT >= idx + 1: break
            return (_sumT + _sumF - 1)

        left[find_index(left,j_seq[i])] = False
        #print np.sum(left)



    #print 'index', find_index(np.array([True, True, False, True]), 0)

    # this is the subgradient part of the projection


    def full_projection(state, n=n, p=p, k=k):
        """
        """
        new_state = np.empty(state.shape, np.float)
        new_state[:n] = state[:n]
        curr = n
        for i in range(k):
            projection = projection_cone(p-i, j_seq[i], s_seq[i])
            new_state[curr:(curr+p-i)] = projection(state[curr:(curr+p-i)])
            curr = curr+p-i
        return new_state

    def full_gradient(state, n=n, p=p, k=k, X=X, mat=mat):
        data = state[:n]

        grad = np.empty(n + np.sum([i for i in range(p-k+1,p+1)]))
        grad[:n] = - data

        curr = n
        for i in range(k):
            subgrad = state[curr:(curr+p-i)]

            sign_vec = np.sign(-mat[i].T.dot(data) + subgrad)
            grad[curr:(curr + p - i)] = -sign_vec
            curr = curr+p-i
            grad[:n] += mat[i].dot(sign_vec)

        return grad



    sampler = projected_langevin(state,
                                 full_gradient,
                                 full_projection,
                                 1./p)
    samples = []

    for _ in range(5000):
        sampler.next()
        samples.append(sampler.state.copy())

    samples = np.array(samples)
    Z = samples[:,:n]

    print 'matk', mat[k-1].shape
    pop = np.abs(mat[k-1].T.dot(Z.T)).max(0)
    fam = discrete_family(pop, np.ones_like(pop))
    pval = fam.cdf(0, obs)
    pval = 2 * min(pval, 1 - pval)

    #stop

    print 'pvalue:', pval
    return pval


if __name__ == "__main__":
    P0 = []
    for i in range(50):
        print "iteration", i
        # print form_Ab(1,4)
        pval = test_2fstep()
        P0.append(pval)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=True)
    plt.show()


