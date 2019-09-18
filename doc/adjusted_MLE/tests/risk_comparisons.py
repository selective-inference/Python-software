import numpy as np, sys, time

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.algorithms.lasso import ROSI

def relative_risk(est, truth, Sigma):
    if (truth != 0).sum > 0:
        return (est - truth).T.dot(Sigma).dot(est - truth) / truth.T.dot(Sigma).dot(truth)
    else:
        return (est - truth).T.dot(Sigma).dot(est - truth)

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y, lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)

                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)

    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    estimate_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.1se')))
    return estimate, estimate_1se, estimate_min, lam_min, lam_1se

def risk_comparison(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                    randomizer_scale=np.sqrt(0.50), full_dispersion=False,
                    tuning_nonrand="lambda.min", tuning_rand="lambda.1se",
                    ndraw =50):

    risks = np.zeros((6,1))
    for i in range(ndraw):
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        print("snr", snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()

        if full_dispersion:
            print("shapes", y.shape, (np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2).shape)
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            _sigma_ = np.std(y)
        lam_theory = _sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        glm_LASSO_theory, glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory / float(n))

        if full_dispersion is False:
            dispersion = None
            active_min = (glm_LASSO_min!=0)
            if active_min.sum()>0:
                sigma_ = np.sqrt(np.linalg.norm(y - X[:,active_min].dot(np.linalg.pinv(X[:,active_min]).dot(y))) ** 2
                                 / (n - active_min.sum()))
            else:
                sigma_ = _sigma_

        print("true and estimated sigma", sigma, _sigma_, sigma_)

        if tuning_nonrand == "lambda.min":
            lam_LASSO = lam_min
            glm_LASSO = glm_LASSO_min
        elif tuning_nonrand == "lambda.1se":
            lam_LASSO = lam_1se
            glm_LASSO = glm_LASSO_1se
        else:
            lam_LASSO = lam_theory / float(n)
            glm_LASSO = glm_LASSO_theory
        active_LASSO = (glm_LASSO != 0)
        rel_LASSO = np.zeros(p)
        if active_LASSO.sum() > 0:
            post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            rel_LASSO[active_LASSO] = post_LASSO_OLS

        if tuning_rand == "lambda.min":
            randomized_lasso = lasso.gaussian(X,
                                              y,
                                              feature_weights=n * lam_min * np.ones(p),
                                              randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
        elif tuning_rand == "lambda.1se":
            randomized_lasso = lasso.gaussian(X,
                                              y,
                                              feature_weights=n * lam_1se * np.ones(p),
                                              randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
        else:
            randomized_lasso = lasso.gaussian(X,
                                              y,
                                              feature_weights=lam_theory * np.ones(p),
                                              randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
        signs = randomized_lasso.fit()
        nonzero = signs != 0
        sel_MLE = np.zeros(p)
        ind_est = np.zeros(p)
        randomized_lasso_est = np.zeros(p)
        randomized_rel_lasso_est = np.zeros(p)

        if nonzero.sum() > 0:
            target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(randomized_lasso.loglike,
                                              randomized_lasso._W,
                                              nonzero,
                                              dispersion=dispersion)

            MLE_estimate, _, _, _, _, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                              cov_target,
                                                                                              cov_target_score,
                                                                                              alternatives)
            sel_MLE[nonzero] = MLE_estimate
            ind_est[nonzero] = ind_unbiased_estimator
            randomized_lasso_est = randomized_lasso.initial_soln
            randomized_rel_lasso_est = randomized_lasso._beta_full

        risks += np.vstack((relative_risk(sel_MLE, beta, Sigma),
                            relative_risk(ind_est, beta, Sigma),
                            relative_risk(randomized_lasso_est, beta, Sigma),
                            relative_risk(randomized_rel_lasso_est, beta, Sigma),
                            relative_risk(rel_LASSO, beta, Sigma),
                            relative_risk(glm_LASSO, beta, Sigma)))
        print("risks so far", risks/(i+1))

    return risks/ndraw

