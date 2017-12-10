from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF
from rpy2.robjects.packages import importr
from rpy2 import robjects
from scipy.stats import t as tdist
import statsmodels.api as sm

glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

def glmnet_sigma(X, y):
    robjects.r('''
                glmnet_cv = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)

                out = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
                lam_1se = out$lambda.1se
                return(lam_1se)
                }''')

    try:
        lambda_cv_R = robjects.globalenv['glmnet_cv']
        n, p = X.shape
        r_X = robjects.r.matrix(X, nrow=n, ncol=p)
        r_y = robjects.r.matrix(y, nrow=n, ncol=1)

        lam_1se = lambda_cv_R(r_X, r_y)
        return lam_1se*n
    except:
        return 0.75 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

def boot_lasso_approx_var(n=100, p=50, s=5, signal=5., B=1000, lam_frac=1., randomization_scale=0.7, sigma= 1.):

    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.35, signal=signal, sigma=sigma,
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

def boot_pivot_approx_var(n=100, p=50, s=5, signal=5., B=1000, lam_frac=1., randomization_scale=0.7, sigma= 1.):

    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.35, signal=signal, sigma=sigma,
                                                       random_signs=True, equicorrelated=False)
        n, p = X.shape

        if p>n:
            sigma_est = np.std(y)/2.
            print("sigma est", sigma_est)
        else:
            ols_fit = sm.OLS(y, X).fit()
            sigma_est = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1.)
            print("sigma est", sigma_est)

        #lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_est
        lam = glmnet_sigma(X, y)

        loss = rr.glm.gaussian(X, y)
        epsilon = 1./np.sqrt(n)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale, sigma=sigma_est)

        M_est.solve_map()
        active = M_est._overall

        true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        nactive = np.sum(active)
        print("number of variables selected by randomized LASSO", nactive)

        coverage = np.zeros(nactive)

        if nactive > 0:
            approx_MLE, var, mle_map, _, _, _ = solve_UMVU(M_est.target_transform,
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
                #sys.stderr.write("bootstrap sample" + str(b) + "\n")

            boot_std = boot_pivot.std(0)
            for j in range(nactive):
                if (approx_MLE[j] - (1.65 * boot_std[j])) <= true_target[j] and true_target[j] <= (approx_MLE[j] + (1.65 * boot_std[j])):
                    coverage[j] += 1
            break

    return boot_pivot.reshape((B*nactive,)), boot_pivot.mean(0).sum()/nactive, boot_pivot.std(0), \
           np.true_divide(approx_MLE - true_target, boot_pivot.std(0)), (approx_MLE - true_target).sum() / float(nactive),\
           coverage.sum() / float(nactive)

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

    ndraw = 100
    bias = 0.
    pivot_obs_info = []
    coverage = 0.

    for i in range(ndraw):
        approx = boot_pivot_approx_var(n=4000, p=2000, s=20, signal=5., B=1200)
        if approx is not None:
            pivot_boot = approx[3]
            bias += approx[4]
            coverage += approx[5]

            for j in range(pivot_boot.shape[0]):
                pivot_obs_info.append(pivot_boot[j])

        sys.stderr.write("iteration completed" + str(i) + "\n")
        sys.stderr.write("overall_bias" + str(bias / float(i + 1)) + "\n")
        sys.stderr.write("overall coverage" + str(coverage / float(i + 1)) + "\n")

    # plt.clf()
    # ecdf_boot = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
    # grid = np.linspace(0, 1, 101)
    # print("ecdf", ecdf_boot(grid))
    # plt.plot(grid, ecdf_boot(grid), c='blue', marker='^')
    # plt.plot(grid, grid, 'k--')
    # #plt.show()
    # plt.savefig("/Users/snigdhapanigrahi/Desktop/Boot_pivot_n2000_p4000_amp3.5_rho_0.2_sigma1.png")