from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from selection.randomized.M_estimator import M_estimator
import statsmodels.api as sm
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

def relative_risk(est, truth, Sigma):

    return (est-truth).T.dot(Sigma).dot(est-truth)/truth.T.dot(Sigma).dot(truth)

def AR1(rho, p):
    idx = np.arange(p)
    cov = rho ** np.abs(np.subtract.outer(idx, idx))
    return cov, np.linalg.cholesky(cov)

def risk_selective_mle(n=500, p=100, s=5, signal=5., lam_frac=1., randomization_scale=np.sqrt(0.1)):

    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.35, signal=signal, sigma=1.,
                                                       random_signs=True, equicorrelated=False)
        n, p = X.shape

        if p>n:
            sigma_est = np.std(y)/2.
            print("sigma est", sigma_est)
        else:
            ols_fit = sm.OLS(y, X).fit()
            sigma_est = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1.)
            print("sigma est", sigma_est)

        #sigma_est = 1.
        snr = (beta.T).dot(X.T.dot(X)).dot(beta)/n
        print("snr", snr)

        #lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_est
        lam = glmnet_sigma(X, y)

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
        print("number of variables selected by randomized LASSO", nactive)

        if nactive > 0:
            approx_MLE, var, mle_map, _, _, mle_transform = solve_UMVU(M_est.target_transform,
                                                                       M_est.opt_transform,
                                                                       M_est.target_observed,
                                                                       M_est.feasible_point,
                                                                       M_est.target_cov,
                                                                       M_est.randomizer_precision)

            mle_target_lin, mle_soln_lin, mle_offset = mle_transform
            break

    est_Sigma = X[:, active].T.dot(X[:, active])
    ind_est = mle_target_lin.dot(M_est.target_observed) + mle_soln_lin.dot(M_est.observed_opt_state[:nactive]) + mle_offset
    target_par = beta[active]
    Lasso_est = M_est.observed_opt_state[:nactive]

    return (approx_MLE - target_par).sum()/float(nactive), \
           relative_risk(approx_MLE, target_par, est_Sigma),\
           relative_risk(M_est.target_observed, target_par, est_Sigma),\
           relative_risk(ind_est, target_par, est_Sigma),\
           relative_risk(Lasso_est, target_par, est_Sigma)

def risk_selective_mle_full(n=500, p=100, s=5, signal=5., lam_frac=1., randomization_scale=0.7):

    while True:
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.35, signal=signal, sigma=1.,
                                                       random_signs=True, equicorrelated=False)
        n, p = X.shape

        if p>n:
            sigma_est = np.std(y)/2.
            #sigma_est = 1.
            print("sigma est", sigma_est)
        else:
            ols_fit = sm.OLS(y, X).fit()
            sigma_est = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1.)
            print("sigma est", sigma_est)

        snr = (beta.T).dot(X.T.dot(X)).dot(beta)/n
        print("snr", snr)

        #lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_est
        lam = glmnet_sigma(X, y)

        loss = rr.glm.gaussian(X, y)
        epsilon = 1. /np.sqrt(n)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale, sigma = sigma_est)

        M_est.solve_map()
        active = M_est._overall

        #true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        nactive = np.sum(active)
        print("number of variables selected by randomized LASSO", nactive)

        if nactive > 0:
            approx_MLE, var, mle_map, _, _, mle_transform = solve_UMVU(M_est.target_transform,
                                                                       M_est.opt_transform,
                                                                       M_est.target_observed,
                                                                       M_est.feasible_point,
                                                                       M_est.target_cov,
                                                                       M_est.randomizer_precision)

            mle_target_lin, mle_soln_lin, mle_offset = mle_transform
            break

    Sigma, _ = AR1(rho=0.35, p=p)
    ind_est = np.zeros(p)
    ind_est[active] = mle_target_lin.dot(M_est.target_observed) + mle_soln_lin.dot(M_est.observed_opt_state[:nactive]) + mle_offset
    target_par = beta

    Lasso_est = np.zeros(p)
    Lasso_est[active] = M_est.observed_opt_state[:nactive]
    selective_MLE = np.zeros(p)
    selective_MLE[active] = approx_MLE

    relaxed_Lasso = np.zeros(p)
    relaxed_Lasso[active] = M_est.target_observed

    M_est_nonrand = M_estimator(loss, epsilon, penalty, randomization.isotropic_gaussian((p,), scale=0.005))
    M_est_nonrand.solve()
    rel_Lasso_nonrand = np.zeros(p)
    rel_Lasso_nonrand[M_est_nonrand._overall] = M_est_nonrand.observed_internal_state[M_est_nonrand._overall.sum()]
    Lasso_nonrand = np.zeros(p)
    Lasso_nonrand[M_est_nonrand._overall] = M_est_nonrand.observed_opt_state[:M_est_nonrand._overall.sum()]

    print("number of variables selected by non-randomized LASSO", M_est_nonrand._overall.sum())

    return (selective_MLE - target_par).sum()/float(nactive), \
           relative_risk(selective_MLE, target_par, Sigma), \
           relative_risk(relaxed_Lasso, target_par, Sigma), \
           relative_risk(ind_est, target_par, Sigma), \
           relative_risk(Lasso_est, target_par, Sigma), \
           relative_risk(rel_Lasso_nonrand, target_par, Sigma),\
           relative_risk(Lasso_nonrand, target_par, Sigma)

if __name__ == "__main__":

    ndraw = 100
    bias = 0.
    risk_selMLE = 0.
    risk_relLASSO = 0.
    risk_indest = 0.
    risk_LASSO = 0.
    risk_relLASSO_nonrand = 0.
    risk_LASSO_nonrand = 0.
    for i in range(ndraw):
        approx = risk_selective_mle_full(n=200, p=1000, s=10, signal=3.)
        if approx is not None:
            bias += approx[0]
            risk_selMLE += approx[1]
            risk_relLASSO += approx[2]
            risk_indest += approx[3]
            risk_LASSO += approx[4]
            risk_relLASSO_nonrand += approx[5]
            risk_LASSO_nonrand += approx[6]

        sys.stderr.write("iteration completed" + str(i) + "\n")
        sys.stderr.write("overall_bias" + str(bias / float(i + 1)) + "\n")
        sys.stderr.write("overall_selrisk" + str(risk_selMLE / float(i + 1)) + "\n")
        sys.stderr.write("overall_relLASSOrisk" + str(risk_relLASSO / float(i + 1)) + "\n")
        sys.stderr.write("overall_indepestrisk" + str(risk_indest / float(i + 1)) + "\n")
        sys.stderr.write("overall_LASSOrisk" + str(risk_LASSO / float(i + 1)) + "\n")
        sys.stderr.write("overall_relLASSOrisk_norand" + str(risk_relLASSO_nonrand / float(i + 1)) + "\n")
        sys.stderr.write("overall_LASSOrisk_norand" + str(risk_LASSO_nonrand / float(i + 1)) + "\n")

