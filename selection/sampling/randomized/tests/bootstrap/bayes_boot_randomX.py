import numpy as np
from scipy.stats import laplace, probplot, uniform

#from selection.algorithms.lasso import instance
from instances import instance, bootstrap_covariance
import selection.sampling.randomized.api as randomized
from pvalues_bayes_randomX import pval
#from pvalues_bayes_ranX_gn import pval
from matplotlib import pyplot as plt
import regreg.api as rr
import selection.sampling.randomized.losses.lasso_randomX as lasso_randomX
import statsmodels.api as sm


def test_lasso(s=0, n=100, p=20, weights = "normal",
               randomization_dist = "logistic", randomization_scale = 1,
               Langevin_steps = 10000, burning = 2000, X_scaled = True,
               covariance_estimate = "nonparametric", noise = "uniform"):

    """ weights: exponential, gamma, normal, gumbel
    randomization_dist: logistic, laplace """

    step_size = 1./p

    X, y, true_beta, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0, scale=X_scaled, noise=noise)
    print 'true beta', true_beta
    lam_frac = 1.

    if randomization_dist == "laplace":
        randomization = laplace(loc=0, scale=1.)
        random_Z = randomization.rvs(p)
    if randomization_dist == "logistic":
        random_Z = np.random.logistic(loc=0, scale = 1, size = p)
    if randomization_dist== "normal":
        random_Z = np.random.standard_normal(p)

    print 'randomization', random_Z*randomization_scale
    loss = lasso_randomX.lasso_randomX(X, y)

    epsilon = 1./np.sqrt(n)
    #epsilon = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))+randomization_scale*np.random.logistic(size=(p,10000))).max(0))

    lam_scaled = lam.copy()
    random_Z_scaled = random_Z.copy()
    epsilon_scaled = epsilon

    if (X_scaled == False):
        random_Z_scaled *= np.sqrt(n)
        lam_scaled *= np.sqrt(n)
        epsilon_scaled *= np.sqrt(n)

    penalty = randomized.selective_l1norm_lan(p, lagrange=lam_scaled)

    # initial solution

    problem = rr.simple_problem(loss, penalty)

    random_term = rr.identity_quadratic(epsilon_scaled, 0, -randomization_scale*random_Z_scaled, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    print 'initial solution', initial_soln

    active = (initial_soln != 0)
    if np.sum(active)==0:
        return [-1], [-1]
    inactive = ~active
    betaE = initial_soln[active]
    signs = np.sign(betaE)

    initial_grad = -np.dot(X.T, y - np.dot(X, initial_soln))
    if (X_scaled==False):
        initial_grad /= np.sqrt(n)
    print 'initial_gradient', initial_grad
    subgradient = random_Z - initial_grad - epsilon * initial_soln
    cube = np.divide(subgradient[inactive], lam)

    nactive = betaE.shape[0]
    ninactive = cube.shape[0]

    beta_unpenalized = np.linalg.lstsq(X[:, active], y)[0]
    print 'beta_OLS onto E', beta_unpenalized
    obs_residuals = y - np.dot(X[:, active], beta_unpenalized)  # y-X_E\bar{\beta}^E
    N = np.dot(X[:, inactive].T, obs_residuals)  # X_{-E}^T(y-X_E\bar{\beta}_E), null statistic
    full_null = np.zeros(p)
    full_null[nactive:] = N

    # parametric coveriance estimate
    if covariance_estimate == "parametric":
        XE_pinv = np.linalg.pinv(X[:, active])
        mat = np.zeros((nactive+ninactive, n))
        mat[:nactive,:] = XE_pinv
        mat[nactive:,:] = X[:, inactive].T.dot(np.identity(n)-X[:, active].dot(XE_pinv))
        Sigma_full = mat.dot(mat.T)
    else:
        Sigma_full = bootstrap_covariance(X,y,active, beta_unpenalized)


    init_vec_state = np.zeros(n+nactive+ninactive)
    if weights =="exponential":
        init_vec_state[:n] = np.ones(n)
    else:
        init_vec_state[:n] = np.zeros(n)

    #init_vec_state[:n] = np.random.standard_normal(n)
    #init_vec_state[:n] = np.ones(n)
    init_vec_state[n:(n+nactive)] = betaE
    init_vec_state[(n+nactive):] = cube


    def full_projection(vec_state, signs = signs,
                        nactive=nactive, ninactive = ninactive):

        alpha = vec_state[:n].copy()
        betaE = vec_state[n:(n+nactive)].copy()
        cube = vec_state[(n+nactive):].copy()

        projected_alpha = alpha.copy()
        projected_betaE = betaE.copy()
        projected_cube = np.zeros_like(cube)

        if weights == "exponential":
            projected_alpha = np.clip(alpha, 0, np.inf)

        if weights == "gamma":
            projected_alpha = np.clip(alpha, -2+1./n, np.inf)
        for i in range(nactive):
            if (projected_betaE[i] * signs[i] < 0):
                projected_betaE[i] = 0

        projected_cube = np.clip(cube, -1, 1)

        return np.concatenate((projected_alpha, projected_betaE, projected_cube), 0)


    Sigma = np.linalg.inv(np.dot(X[:, active].T, X[:, active]))
    null, alt = pval(init_vec_state, full_projection, X, obs_residuals, beta_unpenalized, full_null,
                     signs, lam, epsilon,
                     nonzero, active, Sigma,
                     weights, randomization_dist, randomization_scale,
                     Langevin_steps, step_size, burning,
                     X_scaled)
                   #  Sigma_full[:nactive, :nactive])

    return null, alt

if __name__ == "__main__":

    np.random.seed(1)

    plt.figure()
    plt.ion()
    P0, PA = [], []
    for i in range(50):
        print "iteration", i
        p0, pA = test_lasso()
        if np.sum(p0)>-1:
            P0.extend(p0); PA.extend(pA)
            plt.clf()
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            ecdf = sm.distributions.ECDF(P0)
            x = np.linspace(min(P0), max(P0))
            y = ecdf(x)
            plt.plot(x, y, lw=2)
            plt.plot([0, 1], [0, 1], 'k-', lw=1)
        #probplot(P0, dist=uniform, sparams=(0, 1), plot=plt,fit=False)
        #plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
            plt.pause(0.01)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)

    while True:
        plt.pause(0.05)
    plt.savefig('bayes.pdf')

