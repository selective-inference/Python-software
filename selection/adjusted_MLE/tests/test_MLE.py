from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF


def test_lasso(n=100, p=50, s=5, signal=5., B=500, seed_n=0, lam_frac=1., randomization_scale=1.):
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
    n, p = X.shape
    if p>1:
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    else:
        lam = 2.

    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)
    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
    M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale)

    M_est.solve_map()
    active = M_est._overall

    true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
    # true_target = beta[active]
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    if nactive > 0:

        approx_MLE, value, var, mle_map = solve_UMVU(M_est.target_transform,
                                                     M_est.opt_transform,
                                                     M_est.target_observed,
                                                     M_est.feasible_point,
                                                     M_est.target_cov,
                                                     M_est.randomizer_precision)

        boot_sample = np.zeros((B, nactive))
        beta_obs = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(y)
        resid = y - X[:, active].dot(beta_obs)
        for b in range(B):
            boot_indices = np.random.choice(n, n, replace=True)
            boot_vector = (X[boot_indices, :][:, active]).T.dot(resid[boot_indices])
            target_boot = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(boot_vector) + beta_obs
            boot_sample[b, :] = mle_map(target_boot)[0]

        print("estimated sd", boot_sample.std(0))
        return np.true_divide((approx_MLE - true_target), boot_sample.std(0)), (
        (approx_MLE - true_target).sum()) / float(nactive)

    else:
        return None

def test_lasso_approx_var(n=100, p=50, s=5, signal=5., lam_frac=1., randomization_scale=1.):


    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
        n, p = X.shape
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

        loss = rr.glm.gaussian(X, y)
        epsilon = 1./np.sqrt(n)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale)

        M_est.solve_map()
        active = M_est._overall

        true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        nactive = np.sum(active)

        if nactive > 0:
            approx_MLE, value, var, mle_map = solve_UMVU(M_est.target_transform,
                                                         M_est.opt_transform,
                                                         M_est.target_observed,
                                                         M_est.feasible_point,
                                                         M_est.target_cov,
                                                         M_est.randomizer_precision)

            print("approx_MLE", approx_MLE)
            break



    return (approx_MLE - true_target)/np.sqrt(np.diag(var)), (approx_MLE - true_target).sum()/float(nactive)

def orthogonal_lasso_approx(n=100, p=5, s=3, signal=3, lam_frac=1., randomization_scale=1.):

    while True:
        beta = np.zeros(p)

        signal = np.atleast_1d(signal)
        if signal.shape == (1,):
            beta[:s] = signal[0]
        else:
            beta[:s] = np.linspace(signal[0], signal[1], s)

        X = np.identity(n)[:,:p]
        sigma = 1.
        y = (X.dot(beta) + sigma* np.random.standard_normal(n))

        #lam = 2.
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        loss = rr.glm.gaussian(X, y)
        epsilon = 0.
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale)

        M_est.solve_map()
        active = M_est._overall

        nactive = np.sum(active)

        if nactive >0:
            true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
            print("true_target", true_target)
            approx_MLE, value, var, mle_map = solve_UMVU(M_est.target_transform,
                                                         M_est.opt_transform,
                                                         M_est.target_observed,
                                                         M_est.feasible_point,
                                                         M_est.target_cov,
                                                         M_est.randomizer_precision)
            print("approx sd", np.sqrt(np.diag(var)), approx_MLE)
            break

    return np.true_divide((approx_MLE - true_target),np.sqrt(np.diag(var))), (approx_MLE - true_target).sum() / float(nactive)


def test_bias_lasso(nsim=500):
    bias = 0
    for _ in range(nsim):
        bias += test_lasso(n=100, p=50, s=5, signal=5., seed_n=0, lam_frac=1., randomization_scale=1.)[0]

    print(bias / nsim)


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     ndraw = 100
#     boot_pivot = []
#     bias = 0.
#     for i in range(ndraw):
#         boot = test_lasso(n=300, p=1, s=1, signal=5., B=1000, seed_n=i)
#         if boot is not None:
#             pivot = boot[0]
#             bias += boot[1]
#             for j in range(pivot.shape[0]):
#                 boot_pivot.append(pivot[j])
#
#         sys.stderr.write("iteration completed" + str(i) + "\n")
#         sys.stderr.write("overall_bias" + str(bias / float(ndraw)) + "\n")
#         if i % 10 == 0:
#             plt.clf()
#             ecdf = ECDF(ndist.cdf(np.asarray(boot_pivot)))
#             grid = np.linspace(0, 1, 101)
#             print("ecdf", ecdf(grid))
#             plt.plot(grid, ecdf(grid), c='red', marker='^')
#             plt.plot(grid, grid, 'k--')

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     ndraw = 500
#     bias = 0.
#     pivot_obs_info= []
#     for i in range(ndraw):
#         approx = test_lasso_approx_var(n=300, p=50, s=5, signal=0.)
#         if approx is not None:
#             pivot = approx[0]
#             bias += approx[1]
#             for j in range(pivot.shape[0]):
#                 pivot_obs_info.append(pivot[j])
#
#         sys.stderr.write("iteration completed" + str(i) + "\n")
#         sys.stderr.write("overall_bias" + str(bias / float(i)) + "\n")
#
#     #if i % 10 == 0:
#     plt.clf()
#     ecdf = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
#     grid = np.linspace(0, 1, 101)
#     print("ecdf", ecdf(grid))
#     plt.plot(grid, ecdf(grid), c='red', marker='^')
#     plt.plot(grid, grid, 'k--')
#     plt.show()
#     plt.savefig("/Users/snigdhapanigrahi/Desktop/approx_info_selective_MLE_lasso_p1_amp5.png")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ndraw = 500
    bias = 0.
    pivot_obs_info= []
    for i in range(ndraw):
        approx = orthogonal_lasso_approx(n=300, p=5, s=5, signal=0.)
        if approx is not None:
            pivot = approx[0]
            bias += approx[1]
            print("bias in iteration", approx[1])
            for j in range(pivot.shape[0]):
                pivot_obs_info.append(pivot[j])

        sys.stderr.write("iteration completed" + str(i) + "\n")
        sys.stderr.write("overall_bias" + str(bias / float(i)) + "\n")
    print("pivot", np.asarray(pivot_obs_info))
    plt.clf()
    ecdf = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
    grid = np.linspace(0, 1, 101)
    print("ecdf", ecdf(grid))
    plt.plot(grid, ecdf(grid), c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()
    #plt.savefig("/Users/snigdhapanigrahi/Desktop/approx_info_selective_MLE_lasso_p5_amp5.png")

