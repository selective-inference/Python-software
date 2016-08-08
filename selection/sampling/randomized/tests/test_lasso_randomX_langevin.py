import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues_randomX import pval
from matplotlib import pyplot as plt
import regreg.api as rr
import selection.sampling.randomized.losses.lasso_randomX as lasso_randomX


def test_lasso(s=5, n=200, p=20, covariance_estimate = "nonparametric",
               Langevin_steps = 10000, burning=2000):

    step_size = 1./p

    X, y, true_beta, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0.1)
    #print 'true beta', true_beta
    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = lasso_randomX.lasso_randomX(X, y)

    random_Z = randomization.rvs(p)
    epsilon = 1./np.sqrt(n)
    #epsilon = 1.

    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm_lan(p, lagrange=lam)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    initial_grad = loss.smooth_objective(initial_soln,  mode='grad')
    betaE, cube = penalty.setup_sampling(initial_grad,
                                         initial_soln,
                                         random_Z,
                                         epsilon)

    active = penalty.active_set
    inactive = ~active
    loss.fit_E(active)
    beta_unpenalized = loss._beta_unpenalized
    residual = y - np.dot(X[:, active], beta_unpenalized)  # y-X_E\bar{\beta}^E
    N = np.dot(X[:, inactive].T, residual)  # X_{-E}^T(y-X_E\bar{\beta}_E), null statistic
    data = np.concatenate((beta_unpenalized, N), axis=0)
    ndata = data.shape[0];  nactive = betaE.shape[0];  ninactive = cube.shape[0]


    # non-parametric covariance estimate
    #Sigma_full = loss._Sigma_full
    #Sigma_full_inv = np.linalg.inv(Sigma_full)

    init_vec_state = np.zeros(ndata+nactive+ninactive)
    init_vec_state[:ndata] = data
    init_vec_state[ndata:(ndata+nactive)] = betaE
    init_vec_state[(ndata+nactive):] = cube

    def bootstrap_covariance(X=X, y=y, active=active, beta_unpenalized=beta_unpenalized):
        n, p = X.shape
        nsample = 5000
        nactive = np.sum(active)

        _mean_cum_data = 0
        _cov_data = np.zeros((p, p))

        for _ in range(nsample):
            indices = np.random.choice(n, size=(n,), replace=True)
            y_star = y[indices]
            X_star = X[indices]

            # Z_star = np.dot(X_star.T, y_star - pi(X_star))  # X^{*T}(y^*-X^{*T}_E\bar{\beta}_E)
            Z_star = np.dot(X_star.T, y_star - np.dot(X_star[:,active], beta_unpenalized))

            mat_XEstar = np.linalg.inv(np.dot(X_star[:, active].T, X_star[:, active]))  # (X^{*T}_E X^*_E)^{-1}
            mat_star = np.dot(np.dot(X_star[:, inactive].T, X_star[:, active]), mat_XEstar)
            data_star = np.zeros(p)
            data_star[nactive:] = Z_star[inactive,] - np.dot(mat_star, Z_star[active,])
            data_star[:nactive] = np.dot(mat_XEstar, Z_star[active,])

            _mean_cum_data += data_star
            _cov_data += np.multiply.outer(data_star, data_star)

        _cov_data /= nsample
        _mean_cum_data = _mean_cum_data / nsample
        _cov_data -= np.multiply.outer(_mean_cum_data, _mean_cum_data)

        return _cov_data

    if covariance_estimate=="nonparametric":
        Sigma_full = bootstrap_covariance()
    else:
        # parametric coveriance estimate
        XE_pinv = np.linalg.pinv(X[:, active])
        mat = np.zeros((nactive + ninactive, n))
        mat[:nactive, :] = XE_pinv
        mat[nactive:, :] = X[:, inactive].T.dot(np.identity(n) - X[:, active].dot(XE_pinv))

        Sigma_full = mat.dot(mat.T)
        Sigma_full_inv = np.linalg.inv(Sigma_full)

    Sigma_full_inv = np.linalg.inv(Sigma_full)


    def full_projection(vec_state, penalty=penalty,
                        ndata=ndata, nactive=nactive, ninactive = ninactive):
        data = vec_state[:ndata].copy()
        betaE = vec_state[ndata:(ndata+nactive)]
        cube = vec_state[(ndata+nactive):]

        signs = penalty.signs

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

        gradient = loss.gradient(data, params)
        hessian = loss.hessian

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
        #_gradient[:nactive] = - (np.dot(Sigma_T_inv, data[:nactive]) + np.dot(hessian[:, active].T, sign_vec))
        _gradient[ndata:(ndata + nactive)] = np.dot(A_restricted.T, sign_vec)
        _gradient[(ndata + nactive):] = lam * sign_vec[inactive]

        return _gradient




    null, alt = pval(init_vec_state, full_gradient, full_projection,
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
        probplot(P0, dist=uniform, sparams=(0, 1), plot=plt,fit=False)
        plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
        plt.pause(0.01)


    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.suptitle("LASSO with random X")

    while True:
        plt.pause(0.05)

    plt.show()