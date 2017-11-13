from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF

def test_lasso(n=100, p=50, s=5, signal=5., B= 500, seed_n = 0, lam_frac=1., randomization_scale=1.):
    np.random.seed(seed_n)
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

        boot_sample = np.zeros((B, nactive))
        for b in range(B):
            boot_indices = np.random.choice(n, n, replace=True)
            boot_vector = (X[boot_indices, :]).T.dot(y[boot_indices])
            target_boot = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(boot_vector[active])
            boot_sample[b, :] = mle_map(target_boot)[0]

        print("estimated sd", boot_sample.std(0))
        return np.true_divide((approx_MLE- true_target), boot_sample.std(0))
    else:
        return None

def test_bias_lasso(nsim = 500):

    bias = 0
    for _ in range(nsim):
        bias += test_lasso(n=100, p=50, s=5, signal=5., seed_n = 0, lam_frac=1., randomization_scale=1.)[0]

    print(bias/nsim)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ndraw = 50
    boot_pivot= []
    for i in range(ndraw):
        pivot = test_lasso(n=100, p=50, s=0, signal=5., B= 5000, seed_n = i)
        if pivot is not None:
            for j in range(pivot.shape[0]):
                boot_pivot.append(pivot[j])

        sys.stderr.write("iteration completed" + str(i) + "\n")
    plt.clf()
    ecdf = ECDF(ndist.cdf(np.asarray(boot_pivot)))
    grid = np.linspace(0, 1, 101)
    print("ecdf", ecdf(grid))
    plt.plot(grid, ecdf(grid), c='blue', marker='^')
    plt.plot(grid, grid, c='red', marker='^')
    plt.show()
    #plt.savefig("/Users/snigdhapanigrahi/Desktop/boot_selective_MLE_lasso_p50.png")
