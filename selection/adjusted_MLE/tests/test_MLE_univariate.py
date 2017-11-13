from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF

def boot_lasso(n=100, p=50, s=5, signal=5., B=1000, seed_n = 0, lam_frac=1., randomization_scale=1.):

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
    active = M_est._overall
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")

    true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))

    if nactive > 0:
        boot_sample = np.zeros((B, nactive))
        for k in range(nactive):
            M_est.solve_map_univariate_target(k)
            approx_MLE, value, mle_map = solve_UMVU(M_est.target_transform,
                                                    M_est.opt_transform,
                                                    np.array([M_est.target_observed]),
                                                    M_est.feasible_point,
                                                    M_est.target_cov[k,k],
                                                    M_est.randomizer_precision)

            for b in range(B):
                boot_indices = np.random.choice(n, n, replace=True)
                boot_vector = (X[boot_indices, :]).T.dot(y[boot_indices])
                target_boot = ((np.linalg.inv(X[:, active].T.dot(X[:, active]))).dot(boot_vector[active]))[j]
                boot_sample[b,k] = (mle_map(target_boot))[0]

            sys.stderr.write("iteration completed" + str(k) + "\n")

        centered_boot_sample = boot_sample - boot_sample.mean(0)[None, :]
        std_boot_sample = centered_boot_sample / (boot_sample.std(0)[None, :])

        return std_boot_sample.reshape((B * nactive,))
    else:
        return None

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.clf()
    bootstrap = boot_lasso(n=100, p=50, s=5, signal=5., B=5000, seed_n = 0, lam_frac=1., randomization_scale=1.)
    boot_pivot = bootstrap
    ecdf = ECDF(ndist.cdf(boot_pivot))
    grid = np.linspace(0, 1, 101)
    print("ecdf", ecdf(grid))
    plt.plot(grid, ecdf(grid), c='blue', marker='^')
    #plt.plot(grid, grid, c='red', marker='^')
    plt.show()
    #plt.savefig("/Users/snigdhapanigrahi/selective_mle/Plots/only_boot_selective_MLE_lasso_p50.png")