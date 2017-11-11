from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from selection.adjusted_MLE.tests.exact_MLE import grad_CGF
from statsmodels.distributions.empirical_distribution import ECDF

def test(n=100, p=1, s=1, signal=5., seed_n = 0, lam_frac=1., randomization_scale=1.):
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
    n, p = X.shape
    np.random.seed(seed_n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
    #randomizer = randomization.gaussian(np.identity(p))
    M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale)

    M_est.solve_map()
    active = M_est._overall
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    if nactive > 0:
        mle = solve_UMVU(M_est.target_transform,
                         M_est.opt_transform,
                         M_est.target_observed,
                         M_est.feasible_point,
                         M_est.target_cov,
                         M_est.randomizer_precision)

        return mle[0], M_est.target_observed, nactive
    else:
        return None

def simple_problem(target_observed=2, n=1, threshold=2, randomization_scale=1.):
    """
    Simple problem: randomizaiton of sd 1 and thresholded at 2 (default args)
    """
    target_observed = np.atleast_1d(target_observed)
    target_transform = (-np.identity(n), np.zeros(n))
    opt_transform = (np.identity(n), np.ones(n) * threshold)
    feasible_point = np.ones(n)
    randomizer_precision = np.identity(n) / randomization_scale ** 2
    target_cov = np.identity(n)

    return solve_UMVU(target_transform,
                      opt_transform,
                      target_observed,
                      feasible_point,
                      target_cov,
                      randomizer_precision)

def bootstrap_simple(n= 100, B=100, true_mean=0., threshold=2.):

    while True:
        Zval = np.random.normal(true_mean, 1, n)
        omega = np.random.normal(0, 1)
        target_Z = (np.sum(Zval) / np.sqrt(n))
        check = target_Z + omega - threshold
        if check>0.:
            break

    approx_MLE, value, mle_map = simple_problem(target_Z, n=1, threshold=2, randomization_scale=1.)

    boot_sample = []
    for b in range(B):
        Zval_boot = np.sum(Zval[np.random.choice(n, n, replace=True)]) / np.sqrt(n)
        boot_sample.append(mle_map(Zval_boot)[0])

    return boot_sample, np.mean(boot_sample), np.std(boot_sample), np.squeeze((boot_sample - np.mean(boot_sample)) / np.std(boot_sample))

# if __name__ == "__main__":
#     n = 1000
#     Zval = np.random.normal(0, 1, n)
#     sys.stderr.write("observed Z" + str(Zval) + "\n")
#     MLE = simple_problem(Zval, n=n, threshold=2, randomization_scale=1.)[0]
#     #print(MLE)
#
#     mu_seq = np.linspace(-6, 6, 200)
#     grad_partition = np.array([grad_CGF(mu, randomization_scale=1., threshold=2) for mu in mu_seq])
#
#     exact_MLE = []
#     for k in range(Zval.shape[0]):
#         mle = mu_seq[np.argmin(np.abs(grad_partition - Zval[k]))]
#         exact_MLE.append(mle)
#
#     np.testing.assert_allclose(MLE, exact_MLE, rtol=2.0)

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     plt.clf()
#     Zval = np.linspace(-5, 5, 51)
#     MLE = np.array([simple_problem(z)[0] for z in Zval])
#
#     mu_seq = np.linspace(-6, 6, 200)
#     grad_partition = np.array([grad_CGF(mu, randomization_scale=1., threshold=2) for mu in mu_seq])
#
#     plt.plot(Zval, MLE, label='+2')
#     plt.plot(grad_partition, mu_seq, 'r--', label='MLE')
#     plt.legend()
#     plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.clf()
    boot_result = bootstrap_simple(n= 100, B=1000, true_mean=0., threshold=2.)
    boot_pivot = boot_result[3]
    print("boot sample", boot_pivot.shape)
    ecdf = ECDF(ndist.cdf(boot_pivot))
    grid = np.linspace(0, 1, 101)
    print("ecdf", ecdf(grid))
    plt.plot(grid, ecdf(grid), c='red', marker='^')
    plt.show()
