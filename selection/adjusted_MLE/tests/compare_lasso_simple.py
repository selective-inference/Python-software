from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF

def test_lasso_approx_var(n=100, p=1, s=0, signal=0., lam_frac=1., randomization_scale=1.):

    lam = 2.
    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
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
        if nactive > 0:
            true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
            print("true target", true_target)
            approx_MLE, value, var, mle_map = solve_UMVU(M_est.target_transform,
                                                         M_est.opt_transform,
                                                         M_est.target_observed,
                                                         M_est.feasible_point,
                                                         M_est.target_cov,
                                                         M_est.randomizer_precision)

            print("approx_MLE", approx_MLE)
            #print("check maps", M_est.opt_transform, M_est.target_transform, M_est.feasible_point, M_est.target_cov,
            #      M_est.randomizer_precision, M_est.target_observed)

            _ , opt_offset = M_est.opt_transform
            target_observed = np.atleast_1d(M_est.target_observed)
            target_transform = (-np.identity(1), np.zeros(1))
            s = np.asscalar(np.sign(opt_offset))
            opt_transform = (s * np.identity(1), np.ones(1) * (s * 2.))
            feasible_point = np.ones(1)
            randomizer_precision = np.identity(1) / randomization_scale ** 2
            target_cov = np.identity(1)
            approx_MLE_0, value_0, var_0, mle_map_0= solve_UMVU(target_transform,
                                                                opt_transform,
                                                                target_observed,
                                                                feasible_point,
                                                                target_cov,
                                                                randomizer_precision)
            break

    return np.squeeze((approx_MLE - true_target)/float(np.sqrt(var))), (approx_MLE - true_target), \
           np.squeeze((approx_MLE_0 - true_target)/float(np.sqrt(var_0))), (approx_MLE_0 - true_target)


def test_approx_var(n=100, p=1, s=0, signal=0., lam_frac=1., randomization_scale=1.):

    lam = 2.
    while True:
        X = np.ones((n, p)) / float(np.sqrt(n))
        n, p = X.shape
        beta = signal
        y = np.random.standard_normal(n)
        y += (beta / np.sqrt(n))
        omega = np.random.standard_normal(1)

        true_target = beta * np.sqrt(n)
        target_observed = y.sum()/float(np.sqrt(n))
        if np.abs(target_observed + omega) > lam :

            target_transform = (-np.identity(1), np.zeros(1))
            s = np.asscalar(np.sign(target_observed + omega))
            opt_transform = (s * np.identity(1), np.ones(1) * (s * 2.))
            feasible_point = np.ones(1)
            randomizer_precision = np.identity(1) / randomization_scale ** 2
            target_cov = np.identity(1)
            approx_MLE_0, value_0, var_0, mle_map_0= solve_UMVU(target_transform,
                                                                opt_transform,
                                                                target_observed,
                                                                feasible_point,
                                                                target_cov,
                                                                randomizer_precision)
            break

    return np.squeeze((approx_MLE_0 - true_target)/float(np.sqrt(var_0))), (approx_MLE_0 - true_target)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ndraw = 400
    pivot_lasso = []
    pivot_simple = []
    diff = 0.
    bias = 0.
    for i in range(ndraw):
        approx = test_lasso_approx_var(n=300, p=1, s=1, signal=5.)
        if approx is not None:
            pivot_lasso.append(approx[0])
            pivot_simple.append(approx[2])
            bias += approx[1]
            #diff += approx[0]-approx[2]
        sys.stderr.write("iteration completed" + str(i) + "\n")
        sys.stderr.write("bias" + str(bias/float(i)) + "\n")
    #sys.stderr.write("diff" + str(diff) + "\n")

    #if i % 10 == 0:
    plt.clf()
    ecdf = ECDF(ndist.cdf(np.asarray(pivot_lasso)))
    ecdf_0 = ECDF(ndist.cdf(np.asarray(pivot_simple)))
    grid = np.linspace(0, 1, 101)
    #print("ecdf", ecdf(grid))
    plt.plot(grid, ecdf(grid), c='red', marker='^')
    plt.plot(grid, ecdf_0(grid), '-b')
    plt.plot(grid, grid, 'k--')
    plt.show()
    #plt.savefig("/Users/snigdhapanigrahi/Desktop/approx_lasso_selective_MLE_lasso_p1_amp5.png")

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     ndraw = 200
#     pivot_simple = []
#     diff = 0.
#     for i in range(ndraw):
#         approx = test_approx_var(n=300, p=1, s=0, signal=0.)
#         print("here")
#         pivot_simple.append(approx[0])
#         sys.stderr.write("iteration completed" + str(i) + "\n")
#
#     #if i % 10 == 0:
#     plt.clf()
#     ecdf = ECDF(ndist.cdf(np.asarray(pivot_simple)))
#     grid = np.linspace(0, 1, 101)
#     plt.plot(grid, ecdf(grid), c='red', marker='^')
#     plt.plot(grid, grid, 'k--')
#     plt.show()