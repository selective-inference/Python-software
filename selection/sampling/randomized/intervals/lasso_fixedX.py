import numpy as np
from scipy.optimize import minimize
from selection.sampling.randomized.tests import test_lasso_fixedX_langevin as test_lasso
import selection.sampling.randomized.api as randomized
from scipy.stats import laplace, probplot, uniform
from selection.algorithms.lasso import instance
import regreg.api as rr


def intervals(n=50, p=10, s=0):

    X, y, true_beta, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1., rho=0)

    n, p = X.shape
    print 'true beta', true_beta
    lam_frac = 1.

    loss = randomized.gaussian_Xfixed(X, y)
    random_Z = np.random.standard_normal(p)
    tau = 1.
    epsilon = 1. / np.sqrt(n)
    # epsilon = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))
    penalty = randomized.selective_l1norm_lan(p, lagrange=lam)
    # initial solution
    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0,
                                        random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    #print initial_soln
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE, cube = penalty.setup_sampling(initial_grad,
                                         initial_soln,
                                         random_Z,
                                         epsilon)
    active = penalty.active_set
    inactive = ~active
    signs = penalty.signs

    nactive = np.sum(active)
    print 'size of the active set', nactive
    if nactive==0: return 0
    ninactive = np.sum(inactive)
    #X = np.array([[1,2], [3,4]])
    #y = np.array([1, 2])
    #active = np.zeros(2, dtype=bool)
    #active[0] = True
    #nactive = np.sum(active)
    #signs = np.array([1])
    #sigma = 1. ; tau = 1. ; epsilon = 1. ; lam = 4.,


    def joint_Gaussian_parameters(X=X, y=y,
                                  active=active, signs =signs,
                                  j=0):
        n, p = X.shape
        nactive = np.sum(active)

        eta =  np.linalg.pinv(X[:,active])[j,:]
        c = np.true_divide(eta, np.linalg.norm(eta)**2)

        A = np.zeros((p,p+1))
        A[:,0] = -np.dot(X.T,c.T)
        A[:, 1:(nactive+1)] = np.dot(X.T, X[:, active])
        A[:nactive,1:(nactive+1)] += epsilon*np.identity(nactive)
        A[nactive:,(nactive+1):] = lam*np.identity(ninactive)

        fixed_part = np.dot(np.identity(n)-np.outer(c, eta), y)

        b = -np.dot(X.T, fixed_part)
        b[:nactive] += lam*signs

        v = np.zeros(p+1)
        v[0] = 1

        Sigma_inv = np.true_divide(np.dot(A.T, A), tau**2) + np.true_divide(np.outer(v,v), sigma**2)
        Sigma_inv_mu = np.true_divide(np.dot(A.T, b), tau**2)

        return Sigma_inv, Sigma_inv_mu

    Sigma_inv, Sigma_inv_mu = joint_Gaussian_parameters()
    Sigma = np.linalg.inv(Sigma_inv)


    def log_selection_probability(param=0, Sigma_inv=Sigma_inv, Sigma_inv_mu=Sigma_inv_mu):
        Sigma_inv_mu_modified = Sigma_inv_mu
        Sigma_inv_mu_modified[0] += np.true_divide(param, sigma**2)

        initial_guess = np.zeros(p+1)
        initial_guess[1:(nactive+1)] = initial_soln[active]
        print 'initial guess', initial_guess
        print initial_guess.shape[0]
        print Sigma_inv_mu_modified.shape[0]

        bounds = ((None, None),)
        for i in range(nactive):
            if signs[i]<0:
                bounds += ((None, 0),)
            else:
                bounds += ((0,None), )
        bounds += ((-1,1),)* ninactive
        print signs
        print bounds
        #print 'fun', minimize(lambda x:x**2, x0=1).fun
        def objective(x):
            return np.inner(x, Sigma_inv.dot(x))/2 - np.inner(Sigma_inv_mu_modified, x) #\
                  # -np.sum(np.log(np.multiply(signs, x[1:(nactive + 1)]))) \
                  # -np.sum(np.log(1 + x[(nactive + 1):])) - np.sum(np.log(1 - x[(nactive + 1):]))

        res = minimize(objective, x0=np.ones(p+1), bounds=bounds)
        print res.fun
        mu = np.dot(Sigma, Sigma_inv_mu_modified)
        return -np.inner(mu, Sigma_inv_mu_modified)/2 - res.fun

    print log_selection_probability()


    return 0

intervals()














