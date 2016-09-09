import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues1 import pval
from matplotlib import pyplot as plt
import regreg.api as rr


def test_lasso(s=1, n=100, p=10):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    print 'sigma', sigma
    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.gaussian_Xfixed(X, y)

    random_Z = randomization.rvs(p)
    epsilon = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm_lan(p, lagrange=lam)

    #sampler1 = randomized.selective_sampler_MH_lan(loss,
    #                                           random_Z,
    #                                           epsilon,
    #                                           randomization,
    #                                          penalty)

    #loss_args = {'mean': np.zeros(n),
    #             'sigma': sigma,
    #             'linear_part':np.identity(y.shape[0]),
    #             'value': 0}

    #sampler1.setup_sampling(y, loss_args=loss_args)
    # data, opt_vars = sampler1.state

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

    data = y.copy()
    active = penalty.active_set
    if (np.sum(active)==0):
        print 'here'
        return [-1], [-1]
    inactive = ~active

    #betaE, cube = opt_vars
    ndata = data.shape[0];  nactive = betaE.shape[0];  ninactive = cube.shape[0]
    init_vec_state = np.zeros(ndata+nactive+ninactive)
    init_vec_state[:ndata] = data
    init_vec_state[ndata:(ndata+nactive)] = betaE
    init_vec_state[(ndata+nactive):] = cube

    def bootstrap_samples(y, P, R):
        nsample = 50
        boot_samples = []
        for _ in range(nsample):
            indices = np.random.choice(n, size=(n,), replace=True)
            y_star = y[indices]
            boot_samples.append(np.dot(P,y)+np.dot(R,y_star-y))

        return boot_samples

   #boot_samples = bootstrap_samples(y)


    def move_data(vec_state, boot_samples,
                   ndata = ndata, nactive = nactive, ninactive = ninactive, loss=loss):

        weights = []

        betaE = vec_state[ndata:(ndata+nactive)]
        cube = vec_state[(ndata+nactive):]
        opt_vars = [betaE, cube]
        params, _, opt_vec = penalty.form_optimization_vector(opt_vars)  # opt_vec=\epsilon(\beta 0)+u, u=\grad P(\beta), P penalty

        for i in range(len(boot_samples)):
            gradient = loss.gradient(boot_samples[i], params)
            weights.append(np.exp(-np.sum(np.abs(gradient + opt_vec))))
        weights /= np.sum(weights)

        #m = max(weights)
        #idx = [i for i, j in enumerate(weights) if j == m][0]
        idx = np.nonzero(np.random.multinomial(1, weights, size=1)[0])[0][0]
        return boot_samples[idx]


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



    def full_gradient(vec_state, loss=loss, penalty =penalty, X=X,
                      lam=lam, epsilon=epsilon, ndata=ndata, active=active, inactive=inactive):
        nactive = np.sum(active); ninactive=np.sum(inactive)

        data = vec_state[:ndata]
        betaE = vec_state[ndata:(ndata + nactive)]
        cube = vec_state[(ndata + nactive):]

        opt_vars = [betaE, cube]
        params , _ , opt_vec = penalty.form_optimization_vector(opt_vars) # opt_vec=\epsilon(\beta 0)+u, u=\grad P(\beta), P penalty

        gradient = loss.gradient(data, params)
        hessian = loss.hessian()

        ndata = data.shape[0]
        nactive = betaE.shape[0]
        ninactive = cube.shape[0]

        sign_vec = - np.sign(gradient + opt_vec)  # sign(w), w=grad+\epsilon*beta+lambda*u

        B = hessian + epsilon * np.identity(nactive + ninactive)
        A = B[:, active]

        _gradient = np.zeros(ndata + nactive + ninactive)
        _gradient[:ndata] = 0 #- (data + np.dot(X, sign_vec))
        _gradient[ndata:(ndata + nactive)] = np.dot(A.T, sign_vec)
        _gradient[(ndata + nactive):] = lam * sign_vec[inactive]

        return _gradient


    null, alt = pval(init_vec_state, full_gradient, full_projection, move_data, bootstrap_samples,
                      X, y, nonzero, active)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(20):
        print "iteration", i
        p0, pA = test_lasso()
        if (sum(pA)>=0):
            P0.extend(p0); PA.extend(pA)

    plt.figure()
    probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=True)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.show()