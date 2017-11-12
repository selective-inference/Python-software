from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF

def test_lasso(n=100, p=50, s=5, signal=5., seed_n = 0, lam_frac=1., randomization_scale=1.):

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
    n, p = X.shape

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
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
    #true_target = beta[active]
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    if nactive > 0:
        approx_MLE, value, mle_map = solve_UMVU(M_est.target_transform,
                                                M_est.opt_transform,
                                                M_est.target_observed,
                                                M_est.feasible_point,
                                                M_est.target_cov,
                                                M_est.randomizer_precision)

        return np.mean(approx_MLE- true_target), approx_MLE, M_est.target_observed, active, X.T.dot(y), \
               np.linalg.inv(X[:, active].T.dot(X[:, active])), mle_map
    else:
        return None

def test_bias_lasso(nsim = 500):

    bias = 0
    for _ in range(nsim):
        bias += test_lasso(n=100, p=50, s=5, signal=5., seed_n = 0, lam_frac=1., randomization_scale=1.)[0]

    print(bias/nsim)

#test_bias_lasso()

def bootstrap_lasso(B=500):
    p = 100
    run_lasso = test_lasso(n=100, p=p, s=0, signal=5., seed_n = 0, lam_frac=1., randomization_scale=1.)

    boot_sample = np.zeros((B,run_lasso[3].sum()))
    for b in range(B):
        boot_vector = (run_lasso[4])[np.random.choice(p, p, replace=True)]
        #print("shape", boot_vector.shape)
        active = run_lasso[3]
        target_boot = (run_lasso[5]).dot(boot_vector[active])
        boot_sample[b, :] = (run_lasso[6](target_boot))[0]

    centered_boot_sample = boot_sample - boot_sample.mean(0)[None, :]
    std_boot_sample = centered_boot_sample/(boot_sample.std(0)[None,:])

    return std_boot_sample.reshape((B * run_lasso[3].sum(),))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.clf()
    boot_pivot = bootstrap_lasso(B=10000)
    ecdf = ECDF(ndist.cdf(boot_pivot))
    grid = np.linspace(0, 1, 101)
    print("ecdf", ecdf(grid))
    plt.plot(grid, ecdf(grid), c='blue', marker='^')
    plt.plot(grid, grid, c='red', marker='^')
    plt.savefig("/Users/snigdhapanigrahi/selective_mle/Plots/boot_selective_MLE_lasso_no_signal.png")
