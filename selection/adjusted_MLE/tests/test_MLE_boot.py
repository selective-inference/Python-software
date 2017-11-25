from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF

def boot_lasso_approx_var(n=100, p=50, s=5, signal=5., B=1000, lam_frac=1., randomization_scale=1., sigma= 1.):

    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=sigma,
                                                       random_signs=True, equicorrelated=False)
        n, p = X.shape
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

        loss = rr.glm.gaussian(X, y)
        epsilon = 1./np.sqrt(n)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale, sigma=sigma)

        M_est.solve_map()
        active = M_est._overall

        true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        nactive = np.sum(active)

        if nactive > 0:
            approx_MLE, var, mle_map = solve_UMVU(M_est.target_transform,
                                                  M_est.opt_transform,
                                                  M_est.target_observed,
                                                  M_est.feasible_point,
                                                  M_est.target_cov,
                                                  M_est.randomizer_precision)

            boot_sample = np.zeros((B, nactive))
            resid = y - X[:, active].dot(M_est.target_observed)
            for b in range(B):
                boot_indices = np.random.choice(n, n, replace=True)
                boot_vector = (X[boot_indices, :][:, active]).T.dot(resid[boot_indices])
                target_boot = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(boot_vector) + M_est.target_observed
                boot_sample[b, :] = mle_map(target_boot)[0]

            print("estimated sd", boot_sample.std(0), np.sqrt(np.diag(var)))
            return np.true_divide((approx_MLE - true_target), boot_sample.std(0)), \
                   ((approx_MLE - true_target).sum()) / float(nactive), \
                   np.true_divide((approx_MLE - true_target), np.sqrt(np.diag(var)))

            break

def boot_pivot_approx_var(n=100, p=50, s=5, signal=5., B=50000, lam_frac=1., randomization_scale=1., sigma= 1.):

    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=sigma,
                                                       random_signs=True, equicorrelated=False)
        n, p = X.shape
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

        loss = rr.glm.gaussian(X, y)
        epsilon = 1./np.sqrt(n)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale, sigma=sigma)

        M_est.solve_map()
        active = M_est._overall

        true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        nactive = np.sum(active)

        if nactive > 0:
            approx_MLE, var, mle_map = solve_UMVU(M_est.target_transform,
                                                  M_est.opt_transform,
                                                  M_est.target_observed,
                                                  M_est.feasible_point,
                                                  M_est.target_cov,
                                                  M_est.randomizer_precision)

            boot_pivot = np.zeros((B, nactive))
            resid = y - X[:, active].dot(M_est.target_observed)
            for b in range(B):
                boot_indices = np.random.choice(n, n, replace=True)
                boot_vector = (X[boot_indices, :][:, active]).T.dot(resid[boot_indices])
                target_boot = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(boot_vector) + M_est.target_observed
                boot_mle = mle_map(target_boot)
                boot_pivot[b, :] = np.true_divide(boot_mle[0]- approx_MLE, np.sqrt(np.diag(boot_mle[1])))

                sys.stderr.write("bootstrap sample" + str(b) + "\n")

            break

    return boot_pivot.reshape((B*nactive,)), boot_pivot.mean(0).sum()/nactive, boot_pivot.std(0)

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     ndraw = 100
#     bias = 0.
#     pivot_obs_info= []
#     pivot_bootstrap = []
#     for i in range(ndraw):
#         approx = boot_lasso_approx_var(n=300, p=50, s=5, signal=3.5)
#         if approx is not None:
#             pivot_boot = approx[0]
#             pivot_approx_info = approx[2]
#             bias += approx[1]
#             for j in range(pivot_boot.shape[0]):
#                 pivot_obs_info.append(pivot_approx_info[j])
#                 pivot_bootstrap.append(pivot_boot[j])
#
#         sys.stderr.write("iteration completed" + str(i) + "\n")
#         sys.stderr.write("overall_bias" + str(bias / float(i+1)) + "\n")
#         #print("pivots", pivot_approx_info, pivot_boot)
#
#     #if i % 10 == 0:
#     plt.clf()
#     ecdf_approx = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
#     ecdf_boot = ECDF(ndist.cdf(np.asarray(pivot_bootstrap)))
#     grid = np.linspace(0, 1, 101)
#     print("ecdf", ecdf_boot(grid))
#     plt.plot(grid, ecdf_approx(grid), c='red', marker='^')
#     plt.plot(grid, ecdf_boot(grid), c='blue', marker='^')
#     plt.plot(grid, grid, 'k--')
#     plt.show()
#     #plt.savefig("/Users/snigdhapanigrahi/Desktop/Boot_pivot_n2000_p2000_amp3.5_sigma1.png")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bias = 0.
    approx = boot_pivot_approx_var(n=1000, p=300, s=20, signal=3.5)
    if approx is not None:
        pivot_boot = approx[0]
        bias = approx[1]

    sys.stderr.write("overall_bias" + str(bias) + "\n")

    plt.clf()
    ecdf_boot = ECDF(ndist.cdf(np.asarray(pivot_boot)))
    grid = np.linspace(0, 1, 101)
    print("ecdf", ecdf_boot(grid))
    plt.plot(grid, ecdf_boot(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()
    #plt.savefig("/Users/snigdhapanigrahi/Desktop/Boot_pivot_n2000_p2000_amp3.5_sigma1.png")