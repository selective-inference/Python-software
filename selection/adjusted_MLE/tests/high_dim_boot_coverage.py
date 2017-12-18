from __future__ import print_function
from rpy2.robjects.packages import importr
from rpy2 import robjects

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import statsmodels.api as sm
import numpy as np, sys
import regreg.api as rr
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from scipy.stats import norm as ndist
from selection.algorithms.debiased_lasso import _find_row_approx_inverse

def glmnet_sigma(X, y):
    robjects.r('''
                glmnet_cv = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                out = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
                lam_1se = out$lambda.1se
                lam_min = out$lambda.min
                return(list(lam_min = n * as.numeric(lam_min), lam_1se = n* as.numeric(lam_1se)))
                }''')

    lambda_cv_R = robjects.globalenv['glmnet_cv']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    lam = lambda_cv_R(r_X, r_y)
    lam_min = np.array(lam.rx2('lam_min'))
    lam_1se = np.array(lam.rx2('lam_1se'))
    return lam_min, lam_1se


def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    source('~/best-subset/bestsubset/R/sim.R')
    ''')

    r_simulate = robjects.globalenv['sim.xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

def inference_approx(n=100, p=1000, nval=100, rho=0.35, s=5, beta_type=2, snr=0.2,
                     randomization_scale=np.sqrt(0.25), target="partial"):
    while True:
        X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        true_mean = X.dot(beta)

        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))

        X_val -= X_val.mean(0)[None, :]
        X_val /= (X_val.std(0)[None, :] * np.sqrt(nval))

        if p > n:
            sigma_est = np.std(y) / 2.
            #sigma_est = np.std(y)
            print("sigma and sigma_est", sigma, sigma_est)
        else:
            ols_fit = sm.OLS(y, X).fit()
            sigma_est = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1.)
            print("sigma and sigma_est", sigma, sigma_est)

        y = y - y.mean()
        y /= sigma_est
        y_val = y_val - y_val.mean()
        y_val /= sigma_est
        true_mean /= sigma_est

        loss = rr.glm.gaussian(X, y)
        epsilon = 1. / np.sqrt(n)
        lam_seq = np.linspace(0.75, 2.75, num=100) * np.mean(
            np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        err = np.zeros(100)
        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M = np.identity(p)
        for k in range(100):
            lam = lam_seq[k]
            W = np.ones(p) * lam
            penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
            M_est = M_estimator_map(loss, epsilon, penalty, randomizer, M, target=target,
                                    randomization_scale=randomization_scale, sigma=1.)

            active = M_est._overall
            nactive = active.sum()
            approx_MLE_est = np.zeros(p)
            if nactive > 0:
                M_est.solve_map()
                approx_MLE = solve_UMVU(M_est.target_transform,
                                        M_est.opt_transform,
                                        M_est.target_observed,
                                        M_est.feasible_point,
                                        M_est.target_cov,
                                        M_est.randomizer_precision)[0]
                approx_MLE_est[active] = approx_MLE

            err[k] = np.mean((y_val - X_val.dot(approx_MLE_est)) ** 2.)

        lam = lam_seq[np.argmin(err)]
        print('lambda', lam)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, M, target=target,
                                randomization_scale=randomization_scale, sigma=1.)
        active = M_est._overall
        nactive = np.sum(active)

        print("number of variables selected by randomized LASSO", nactive)

        if nactive > 0:
            M_est.solve_map()
            approx_MLE, var, mle_map, _, _, mle_transform = solve_UMVU(M_est.target_transform,
                                                                       M_est.opt_transform,
                                                                       M_est.target_observed,
                                                                       M_est.feasible_point,
                                                                       M_est.target_cov,
                                                                       M_est.randomizer_precision)

            approx_sd = np.sqrt(np.diag(var))
            B = 2000
            boot_pivot = np.zeros((B, nactive))
            resid = y - X[:, active].dot(M_est.target_observed)
            for b in range(B):
                boot_indices = np.random.choice(n, n, replace=True)
                boot_vector = (X[boot_indices, :][:, active]).T.dot(resid[boot_indices])
                target_boot = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(boot_vector) + M_est.target_observed
                boot_mle = mle_map(target_boot)
                #print("target_boot", boot_mle[0], approx_MLE)
                boot_pivot[b, :] = np.true_divide(boot_mle[0] - approx_MLE, np.sqrt(np.diag(boot_mle[1])))

            approx_sd = boot_pivot.std(0)

            if nactive == 1:
                approx_MLE = np.array([approx_MLE])
                approx_sd = np.array([approx_sd])

            coverage_sel = 0
            true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(true_mean)
            for j in range(nactive):
                if (approx_MLE[j] - (1.65 * approx_sd[j])) <= true_target[j] and \
                                (approx_MLE[j] + (1.65 * approx_sd[j])) >= true_target[j]:
                    coverage_sel += 1
                print("selective intervals", (approx_MLE[j] - (1.65 * approx_sd[j])),(approx_MLE[j] + (1.65 * approx_sd[j])))

            break

    if True:
        return coverage_sel/float(nactive)

if __name__ == "__main__":

    ndraw = 100
    coverage_sel = 0.

    for i in range(ndraw):
        approx = inference_approx(n=1000, p=2000, nval=500, rho=0.35, s=20, beta_type=2, snr=0.10, target="partial")
        if approx is not None:
            coverage_sel += approx

        sys.stderr.write("selective coverage" + str(coverage_sel / float(i + 1)) + "\n")

        sys.stderr.write("iteration completed" + str(i) + "\n")
