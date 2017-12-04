from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm
from selection.randomized.M_estimator import M_estimator
from rpy2.robjects.packages import importr
from rpy2 import robjects

glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

def glmnet_sigma(X, y):
    robjects.r('''
                glmnet_cv = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)

                out = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
                lam_minCV = out$lambda.min

                coef = coef(out, s = "lambda.min")
                linear.fit = lm(y~ X[, which(coef>0.001)-1])
                sigma_est = summary(linear.fit)$sigma
                return(sigma_est)
                }''')

    try:
        sigma_cv_R = robjects.globalenv['glmnet_cv']
        n, p = X.shape
        r_X = robjects.r.matrix(X, nrow=n, ncol=p)
        r_y = robjects.r.matrix(y, nrow=n, ncol=1)

        sigma_est = sigma_cv_R(r_X, r_y)
        return sigma_est
    except:
        return np.array([1.])


def test_lasso_approx_var(n=100, p=50, s=5, signal=5., lam_frac=1., randomization_scale=1.):

    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.,
                                                       random_signs=True, equicorrelated=False)
        n, p = X.shape

        if p>n:
            sigma_est = glmnet_sigma(X, y)[0]
            print("sigma est", sigma_est)
        else:
            ols_fit = sm.OLS(y, X).fit()
            sigma_est = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1.)
            print("sigma est", sigma_est)

        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_est

        loss = rr.glm.gaussian(X, y)
        epsilon = 1./np.sqrt(n)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale, sigma = sigma_est)

        M_est.solve_map()
        active = M_est._overall

        true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        nactive = np.sum(active)

        if nactive > 0:
            approx_MLE, var, mle_map, _, _ = solve_UMVU(M_est.target_transform,
                                                        M_est.opt_transform,
                                                        M_est.target_observed,
                                                        M_est.feasible_point,
                                                        M_est.target_cov,
                                                        M_est.randomizer_precision)

            print("approx sd", np.sqrt(np.diag(var)))
            break

    return np.true_divide((approx_MLE - true_target), np.sqrt(np.diag(var))), (approx_MLE - true_target).sum()/float(nactive)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ndraw = 100
    bias = 0.
    pivot_obs_info = []
    for i in range(ndraw):
        approx = test_lasso_approx_var(n=500, p=100, s=10, signal=3.5)
        if approx is not None:
            pivot = approx[0]
            bias += approx[1]
            for j in range(pivot.shape[0]):
                pivot_obs_info.append(pivot[j])

        sys.stderr.write("iteration completed" + str(i) + "\n")
        sys.stderr.write("overall_bias" + str(bias / float(i + 1)) + "\n")

    # if i % 10 == 0:
    plt.clf()
    ecdf = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
    grid = np.linspace(0, 1, 101)
    print("ecdf", ecdf(grid))
    plt.plot(grid, ecdf(grid), c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()