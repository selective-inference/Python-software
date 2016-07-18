import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.randomized import logistic_instance
import selection.sampling.randomized.api as randomized
from pvalues_randomX2 import pval
from matplotlib import pyplot as plt
import regreg.api as rr
import selection.sampling.randomized.losses.lasso_randomX as lasso_randomX


def test_lasso(s=5, n=200, p=20):

    # problem setup

    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0)
    print 'true_beta', beta
    nonzero = np.where(beta)[0]
    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.logistic_Xrandom_new(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm_lan_logistic(p, lagrange=lam)

    random_Z = randomization.rvs(p)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)

    active = (initial_soln != 0)
    inactive = ~active
    betaE = initial_soln[active]
    print 'betaE', betaE
    signs = np.sign(betaE)

    w_initial = np.exp(np.dot(X, initial_soln))
    pi_initial = w_initial/(1+w_initial)
    initial_grad = -np.dot(X.T, y-pi_initial)*n/2
    print 'ini', initial_grad
    subgradient = -(initial_grad+epsilon*initial_soln+random_Z)
    cube = subgradient[inactive]/lam
    print 'cube', cube

    initial_grad = loss.smooth_objective(initial_soln,  mode='grad')
    print 'ini', initial_grad
    betaE, cube = penalty.setup_sampling(initial_grad,
                                         initial_soln,
                                         random_Z,
                                         epsilon)

    print 'betaE', betaE
    print 'cube', cube


    active = penalty.active_set
    if (np.sum(active)==0):
        return np.array([-1]), np.array([-1])
    inactive = ~active
    loss.fit_E(active)
    beta_unpenalized = loss._beta_unpenalized
    w = np.exp(np.dot(X[:, active], beta_unpenalized))
    pi = w / (1 + w)
    N = np.dot(X[:, inactive].T, y - pi)
    data = np.concatenate((beta_unpenalized, N), axis=0)

    ndata = data.shape[0];  nactive = betaE.shape[0];  ninactive = cube.shape[0]


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
    Sigma_full_inv = np.linalg.inv(Sigma_full)
    #Sigma_T = Sigma_full[:nactive,:nactive]
    #Sigma_T_inv = np.linalg.inv(Sigma_T)

    # non-parametric covariance estimate
    #Sigma_full = loss._Sigma_full
    #Sigma_full_inv = np.linalg.inv(Sigma_full)


    init_vec_state = np.zeros(ndata+nactive+ninactive)
    init_vec_state[:ndata] = data
    init_vec_state[ndata:(ndata+nactive)] = betaE
    init_vec_state[(ndata+nactive):] = cube


    def full_projection(vec_state, signs=signs,
                        ndata=ndata, nactive=nactive, ninactive = ninactive):

        data = vec_state[:ndata].copy()
        betaE = vec_state[ndata:(ndata+nactive)]
        cube = vec_state[(ndata+nactive):]

        #signs = penalty.signs

        projected_betaE = betaE.copy()
        projected_cube = np.zeros_like(cube)

        for i in range(nactive):
            if (projected_betaE[i] * signs[i] < 0):
                projected_betaE[i] = 0

        projected_cube = np.clip(cube, -1, 1)

        return np.concatenate((data, projected_betaE, projected_cube), 0)


    def full_gradient(vec_state, loss=loss, penalty = penalty, Sigma_full_inv=Sigma_full_inv,
                      lam=lam, epsilon=epsilon, ndata=ndata, active=active, inactive=inactive):
        nactive = np.sum(active); ninactive=np.sum(inactive)

        data = vec_state[:ndata]
        betaE = vec_state[ndata:(ndata + nactive)]
        cube = vec_state[(ndata + nactive):]

        opt_vars = [betaE, cube]
        params , _ , opt_vec = penalty.form_optimization_vector(opt_vars) # opt_vec=\epsilon(\beta 0)+u, u=\grad P(\beta), P penalty

        gradient = loss.gradient(data, params)*n/2
        hessian = loss.hessian * n/2

        ndata = data.shape[0]
        nactive = betaE.shape[0]
        ninactive = cube.shape[0]

        sign_vec = - np.sign(gradient + opt_vec)  # sign(w), w=grad+\epsilon*beta+lambda*u

        A = hessian + epsilon * np.identity(nactive + ninactive)
        A_restricted = A[:, active]

        T = data[:nactive]
        _gradient = np.zeros(ndata + nactive + ninactive)

        # saturated model
        _gradient[:ndata] = - np.dot(Sigma_full_inv, data)
        _gradient[:nactive] -= hessian[:,active].T.dot(sign_vec)
        _gradient[nactive:(ndata)] -= sign_vec[inactive]

        # selected model
        #gradient[:nactive] = - (np.dot(Sigma_T_inv, data[:nactive]) + np.dot(hessian[:, active].T, sign_vec))

        _gradient[ndata:(ndata + nactive)] = np.dot(A_restricted.T, sign_vec)
        _gradient[(ndata + nactive):] = lam * sign_vec[inactive]

        return _gradient


    null, alt = pval(init_vec_state, full_gradient, full_projection,
                      Sigma_full[:nactive, :nactive], data, nonzero, active)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(20):
        print "iteration", i
        p0, pA = test_lasso()
        P0.extend(p0); PA.extend(pA)


    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0,1], color='k', linestyle='-', linewidth=2)
    plt.show()