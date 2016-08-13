import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues_fixedX import pval
from matplotlib import pyplot as plt
import regreg.api as rr


def test_lasso(X, y, nonzero, sigma, random_Z, randomization_distribution, Langevin_steps=10000, burning=2000):

    n, p = X.shape
    step_size = 1./p
    print 'true beta', true_beta
    lam_frac = 1.

    loss = randomized.gaussian_Xfixed(X, y)

    epsilon = 1./np.sqrt(n)
    #epsilon = 1.

    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    penalty = randomized.selective_l1norm_lan(p, lagrange=lam)


    # initial solution
    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0,
                                        -random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    initial_grad = loss.smooth_objective(initial_soln,  mode='grad')
    betaE, cube = penalty.setup_sampling(initial_grad,
                                         initial_soln,
                                         random_Z,
                                         epsilon)

    signs = np.sign(betaE)
    data = y.copy()
    active = penalty.active_set
    inactive = ~active

    #betaE, cube = opt_vars
    ndata = data.shape[0];  nactive = betaE.shape[0];  ninactive = cube.shape[0]
    init_vec_state = np.zeros(n+nactive+ninactive)
    init_vec_state[:ndata] = data
    init_vec_state[ndata:(ndata+nactive)] = betaE
    init_vec_state[(ndata+nactive):] = cube


    def full_projection(vec_state, signs=signs,
                        ndata=ndata, nactive=nactive, ninactive = ninactive):

        data = vec_state[:ndata].copy()
        betaE = vec_state[ndata:(ndata+nactive)]
        cube = vec_state[(ndata+nactive):]

        projected_betaE = betaE.copy()
        projected_cube = np.zeros_like(cube)

        for i in range(nactive):
            if (projected_betaE[i] * signs[i] < 0):
                projected_betaE[i] = 0

        projected_cube = np.clip(cube, -1, 1)

        return np.concatenate((data, projected_betaE, projected_cube), 0)


    hessian = np.dot(X.T,X)

    def full_gradient(vec_state, X=X,
                      lam=lam, epsilon=epsilon, ndata=ndata, active=active, inactive=inactive):

        nactive = np.sum(active); ninactive=np.sum(inactive)
        data = vec_state[:ndata]
        betaE = vec_state[ndata:(ndata + nactive)]
        cube = vec_state[(ndata + nactive):]

        #opt_vars = [betaE, cube]
        #params , _ , opt_vec = penalty.form_optimization_vector(opt_vars) # opt_vec=\epsilon(\beta 0)+u, u=\grad P(\beta), P penalty

        beta_full = np.zeros(p)
        beta_full[active] = betaE
        subgradient = np.zeros(p)
        subgradient[inactive] = lam * cube
        subgradient[active] = lam * signs

        opt_vec = epsilon * beta_full + subgradient
        gradient = -np.dot(X.T,data-np.dot(X,beta_full))

        ndata = data.shape[0]
        nactive = betaE.shape[0]
        ninactive = cube.shape[0]

        omega = gradient+opt_vec

        if randomization_distribution == "laplace":
                randomization_derivative = - np.sign(omega)  # sign(w), w=grad+\epsilon*beta+lambda*u
        if randomization_distribution == "normal":
                randomization_derivative = - omega
        if randomization_distribution == "logistic":
                randomization_derivative = (np.exp(-omega)-1)/(np.exp(-omega)+1)

        B = hessian + epsilon * np.identity(nactive + ninactive)
        A = B[:, active]

        _gradient = np.zeros(ndata + nactive + ninactive)
        _gradient[:ndata] = - data - np.dot(X, randomization_derivative)
        _gradient[ndata:(ndata + nactive)] = np.dot(A.T, randomization_derivative)
        _gradient[(ndata + nactive):] = lam * randomization_derivative[inactive]

        # data_proposal = np.dot(P, data) + np.dot(R, data_proposal)

        return _gradient


    null, alt = pval(init_vec_state, full_gradient, full_projection,
                      X, y, nonzero, active,
                     Langevin_steps, burning, step_size)

    return null, alt

if __name__ == "__main__":

    s = 5; n = 100; p = 20
    randomization_distribution = "normal"

    P0, PA = [], []
    for i in range(50):
        print "iteration", i
        X, y, true_beta, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1., rho=0)

        if randomization_distribution == "laplace":
            randomization = laplace(loc=0, scale=1.)
            random_Z = randomization.rvs(p)
        if randomization_distribution == "normal":
            random_Z = np.random.standard_normal(p)
        if randomization_distribution == "logistic":
            random_Z = np.random.logistic(loc=0, scale=1.)

        p0, pA = test_lasso(X,y, nonzero, sigma, random_Z, randomization_distribution)
        P0.extend(p0); PA.extend(pA)

    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0,1], color='k', linestyle='-', linewidth=2)
    plt.suptitle("LASSO with fixed X")
    plt.show()
