import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues1 import pval
from matplotlib import pyplot as plt
import regreg.api as rr

def test_lasso(s=5, n=200, p=20):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    print 'sigma', sigma
    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.gaussian_Xfixed(X, y)

    random_Z = randomization.rvs(p)
    epsilon = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    random_Z = randomization.rvs(p)
    lam=1
    penalty = randomized.selective_l1norm_lan(p, lagrange=lam)


    # initial solution
    groups = np.arange(p)
    problem = rr.simple_problem(loss, penalty)

    random_term = rr.identity_quadratic(epsilon, 0,
                                        random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)

    initial_grad = loss.smooth_objective(initial_soln,  mode='grad')
    betaE, cube = penalty.setup_sampling(initial_grad,
                                         initial_soln,
                                         random_Z,
                                         epsilon)

    data = y.copy()
    active = penalty.active_set; inactive = ~active
    nactive = np.sum(active); ninactive = np.sum(inactive)

    subgradient_initial = np.dot(X.T, y-np.dot(X, initial_soln)) -random_Z-epsilon*initial_soln
    subgradient = np.zeros(p)
    subgradient[:nactive] = subgradient_initial[active]
    subgradient[nactive:] = subgradient_initial[inactive]

    z_active = subgradient_initial[active]
    #gamma = np.sqrt(np.inner(betaE, betaE))
    gamma = np.abs(betaE)

    data = y.copy()

    ndata = data.shape[0]
    init_vec_state = np.zeros(ndata+nactive+ninactive+nactive)
    init_vec_state[:ndata] = data
    init_vec_state[ndata:(ndata+nactive+ninactive)] = subgradient
    init_vec_state[(ndata+nactive+ninactive):] = gamma


    def full_projection(vec_state, lam = lam,
                        ndata=ndata, nactive=nactive, ninactive = ninactive):
        data = vec_state[:ndata].copy()
        subgradient = vec_state[ndata:(ndata+nactive+ninactive)]
        gamma = vec_state[(ndata+nactive+ninactive):]

        projected_subgradient = subgradient.copy()
        projected_gamma = gamma.copy()

        #projected_subgradient[:nactive] = lam*np.sign(projected_subgradient[:nactive])
        projected_subgradient[nactive:] = np.clip(projected_subgradient[nactive:], -lam, lam)

        projected_gamma = np.clip(projected_gamma, 0, np.inf)

        return np.concatenate((data, projected_subgradient, projected_gamma), 0)


    def full_gradient(vec_state, X=X,
                      lam=lam, epsilon=epsilon, ndata=ndata, active=active, inactive=inactive):

        nactive = np.sum(active); ninactive=np.sum(inactive)

        data = vec_state[:ndata]
        subgradient = vec_state[ndata:(ndata + nactive+ninactive)]
        gamma = vec_state[(ndata + nactive+ninactive):]


        mat = np.dot(X.T, X) + epsilon*np.identity(X.shape[1])
        vec = np.multiply(subgradient[:nactive], gamma)/lam
        w = - (np.dot(mat[:, active], vec)-np.dot(X.T, y)+subgradient)


        B = np.dot(mat[:, active], np.diag(gamma))/lam
        A = np.concatenate((B, np.zeros((nactive+ninactive, ninactive))), 1)
        A = A + np.identity(nactive+ninactive)

        mat_gamma = np.dot(mat[:, active], np.diag(np.diag(subgradient[:nactive])))/lam

        sign_vec = np.sign(w)

        _gradient = np.zeros(ndata + nactive + ninactive+nactive)
        _gradient[:ndata] = - (data + np.dot(X, sign_vec))
        _gradient[ndata:(ndata + nactive+ninactive)] = np.dot(A.T, sign_vec)
        _gradient[(ndata + nactive+ninactive):] = np.dot(mat_gamma.T, sign_vec)

        # data_proposal = np.dot(P, data) + np.dot(R, data_proposal)

        return _gradient


    null, alt = pval(init_vec_state, full_gradient, full_projection,
                      X, y, nonzero, active)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(20):
        print "iteration", i
        p0, pA = test_lasso()
        P0.extend(p0); PA.extend(pA)

    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0,1], color='k', linestyle='-', linewidth=2)
    plt.show()