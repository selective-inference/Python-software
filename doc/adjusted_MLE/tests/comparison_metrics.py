from __future__ import division, print_function
import numpy as np, sys, time
from scipy.stats import norm as ndist

from rpy2 import robjects
import rpy2.robjects.numpy2ri

from ...randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from ...algorithms.lasso import ROSI
from ...tests.instance import gaussian_instance

def BHfilter(pval, q=0.2):
    pval = np.asarray(pval)
    pval_sort = np.sort(pval)
    comparison = q * np.arange(1, pval.shape[0] + 1.) / pval.shape[0]
    passing = pval_sort < comparison
    if passing.sum():
        thresh = comparison[np.nonzero(passing)[0].max()]
        return np.nonzero(pval <= thresh)[0]
    return []

def sim_xy(n, 
           p, 
           nval, 
           rho=0, 
           s=5, 
           beta_type=2, 
           snr=1):
    try:
        rpy2.robjects.numpy2ri.activate()
        robjects.r('''
        #library(bestsubset)
        source('~/best-subset/bestsubset/R/sim.R')
        sim_xy = sim.xy
        ''')

        r_simulate = robjects.globalenv['sim_xy']
        sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
        X = np.array(sim.rx2('x'))
        y = np.array(sim.rx2('y'))
        X_val = np.array(sim.rx2('xval'))
        y_val = np.array(sim.rx2('yval'))
        Sigma = np.array(sim.rx2('Sigma'))
        beta = np.array(sim.rx2('beta'))
        sigma = np.array(sim.rx2('sigma'))
        rpy2.robjects.numpy2ri.deactivate()
        return X, y, X_val, y_val, Sigma, beta, sigma
    except:
        X, y, beta, _, sigma, Sigma = gaussian_instance(n=n,
                                                        p=p,
                                                        s=s,
                                                        signal=snr,
                                                        equicorrelated=False,
                                                        rho=rho)
        X_val = gaussian_instance(n=n,
                                  p=p,
                                  s=s,
                                  signal=snr,
                                  equicorrelated=False,
                                  rho=rho)[0]
        y_val = X_val.dot(beta) + sigma * np.random.standard_normal(X_val.shape[0])
        return X, y, X_val, y_val, Sigma, beta, sigma

def selInf_R(X, y, beta, lam, sigma, Type, alpha=0.1):
    robjects.r('''
               library("selectiveInference")
               selInf = function(X, y, beta, lam, sigma, Type, alpha= 0.1){
               y = as.matrix(y)
               X = as.matrix(X)
               beta = as.matrix(beta)
               lam = as.matrix(lam)[1,1]
               sigma = as.matrix(sigma)[1,1]
               Type = as.matrix(Type)[1,1]
               if(Type == 1){
                   type = "full"} else{
                   type = "partial"}
               inf = fixedLassoInf(x = X, y = y, beta = beta, lambda=lam, family = "gaussian",
                                   intercept=FALSE, sigma=sigma, alpha=alpha, type=type)
               return(list(ci = inf$ci, pvalue = inf$pv))}
               ''')

    inf_R = robjects.globalenv['selInf']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_beta = robjects.r.matrix(beta, nrow=p, ncol=1)
    r_lam = robjects.r.matrix(lam, nrow=1, ncol=1)
    r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)
    r_Type = robjects.r.matrix(Type, nrow=1, ncol=1)
    output = inf_R(r_X, r_y, r_beta, r_lam, r_sigma, r_Type)
    ci = np.array(output.rx2('ci'))
    pvalue = np.array(output.rx2('pvalue'))
    return ci, pvalue


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
                return(list(estimate = estimate, estimate.1se = estimate.1se, 
                            estimate.min = estimate.min, 
                            lam.min = fit.cv$lambda.min, 
                            lam.1se = fit.cv$lambda.1se))
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

def coverage(intervals, pval, target, truth):
    pval_alt = (pval[truth != 0]) < 0.1
    if pval_alt.sum() > 0:
        avg_power = np.mean(pval_alt)
    else:
        avg_power = 0.
    return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), avg_power

def relative_risk(est, truth, Sigma):
    if (truth != 0).sum > 0:
        return (est - truth).T.dot(Sigma).dot(est - truth) / truth.T.dot(Sigma).dot(truth)
    else:
        return (est - truth).T.dot(Sigma).dot(est - truth)


def comparison_cvmetrics_selected(n=500, 
                                  p=100, 
                                  nval=500, 
                                  rho=0.35, 
                                  s=5, 
                                  beta_type=1, 
                                  snr=0.20,
                                  randomizer_scale=np.sqrt(0.50), 
                                  full_dispersion=True,
                                  tuning_nonrand="lambda.min", 
                                  tuning_rand="lambda.1se"):

    (X, y, _, _, Sigma, beta, sigma) = sim_xy(n=n, 
                                              p=p, 
                                              nval=nval, 
                                              rho=rho, 
                                              s=s, 
                                              beta_type=beta_type, 
                                              snr=snr)

    true_mean = X.dot(beta)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = (sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, 
                    np.random.standard_normal((n, 2000)))).max(0)))
    (glm_LASSO_theory, 
     glm_LASSO_1se, 
     glm_LASSO_min, 
     lam_min, 
     lam_1se) = glmnet_lasso(X, y, lam_theory / n)

    if tuning_nonrand == "lambda.min":
        lam_LASSO = lam_min
        glm_LASSO = glm_LASSO_min
    elif tuning_nonrand == "lambda.1se":
        lam_LASSO = lam_1se
        glm_LASSO = glm_LASSO_1se
    else:
        lam_LASSO = lam_theory/float(n)
        glm_LASSO = glm_LASSO_theory
    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.asarray([(np.in1d(active_set_LASSO[z], true_set).sum() > 0) for 
                                    z in range(nactive_LASSO)], np.bool)

    rel_LASSO = np.zeros(p)
    Lee_nreport = 0
    bias_Lee = 0.
    bias_naive = 0.

    if nactive_LASSO > 0:
        post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        rel_LASSO[active_LASSO] = post_LASSO_OLS
        Lee_target = np.linalg.pinv(X[:, active_LASSO]).dot(X.dot(beta))
        Lee_intervals, Lee_pval = selInf_R(X, 
                                           y, 
                                           glm_LASSO, 
                                           n * lam_LASSO, 
                                           sigma_, 
                                           Type=0, 
                                           alpha=0.1)

        if (Lee_pval.shape[0] == Lee_target.shape[0]):

            cov_Lee, selective_Lee_power = coverage(Lee_intervals, 
                                                    Lee_pval, 
                                                    Lee_target, 
                                                    beta[active_LASSO])

            inf_entries_bool = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            inf_entries = np.mean(inf_entries_bool)
            if inf_entries == 1.:
                length_Lee = 0.
            else:
                length_Lee = (np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])
                                      [~inf_entries_bool]))
            power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]), 
                                                              (0. > Lee_intervals[:, 1])))) \
                            .sum() / float((beta != 0).sum())
            Lee_discoveries = BHfilter(Lee_pval, q=0.1)
            power_Lee_BH = ((Lee_discoveries * active_LASSO_bool).sum() / 
                            float((beta != 0).sum()))
            fdr_Lee_BH = ((Lee_discoveries * ~active_LASSO_bool).sum() / 
                           float(max(Lee_discoveries.sum(), 1.)))
            bias_Lee = np.mean(glm_LASSO[active_LASSO] - Lee_target)

            naive_sd = sigma_ * np.sqrt(np.diag(
                    (np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
            naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                         post_LASSO_OLS + 1.65 * naive_sd]).T
            naive_pval = 2 * ndist.cdf(np.abs(post_LASSO_OLS) / naive_sd)

            cov_naive, selective_naive_power = coverage(naive_intervals, 
                                                        naive_pval, 
                                                        Lee_target, 
                                                        beta[active_LASSO])

            length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
            power_naive = ((active_LASSO_bool) * (
                np.logical_or((0. < naive_intervals[:, 0]), 
                              (0. > naive_intervals[:, 1])))).sum() / float(
                (beta != 0).sum())

            naive_discoveries = BHfilter(naive_pval, q=0.1)

            power_naive_BH = ((naive_discoveries * active_LASSO_bool).sum() / 
                              float((beta != 0).sum()))
            fdr_naive_BH = ((naive_discoveries * ~active_LASSO_bool).sum() / 
                            float(max(naive_discoveries.sum(), 1.)))

            bias_naive = np.mean(rel_LASSO[active_LASSO] - Lee_target)

            partial_Lasso_risk = (glm_LASSO[active_LASSO]-Lee_target).T.dot(
                                  glm_LASSO[active_LASSO]-Lee_target)
            partial_relLasso_risk = (post_LASSO_OLS - Lee_target).T.dot(
                                     post_LASSO_OLS - Lee_target)

        else:
            Lee_nreport = 1
            (cov_Lee, 
             length_Lee, 
             inf_entries, 
             power_Lee, 
             power_Lee_BH, 
             fdr_Lee_BH, 
             selective_Lee_power) = [0., 0., 0., 0., 0., 0., 0.]

            (cov_naive, 
             length_naive, 
             power_naive, 
             power_naive_BH, 
             fdr_naive_BH, 
             selective_naive_power) = [0., 0., 0., 0., 0., 0.]

            naive_discoveries = np.zeros(1)
            Lee_discoveries = np.zeros(1)
            partial_Lasso_risk,  partial_relLasso_risk = [0., 0.]

    elif nactive_LASSO == 0:
        Lee_nreport = 1
        (cov_Lee, 
         length_Lee, 
         inf_entries, 
         power_Lee, 
         power_Lee_BH, 
         fdr_Lee_BH, 
         selective_Lee_power) = [0., 0., 0., 0., 0., 0., 0.]

        (cov_naive, 
         length_naive, 
         power_naive, 
         power_naive_BH, 
         fdr_naive_BH, 
         selective_naive_power) = [0., 0., 0., 0., 0., 0.]

        naive_discoveries = np.zeros(1)
        Lee_discoveries = np.zeros(1)
        partial_Lasso_risk, partial_relLasso_risk = [0., 0.]

    if tuning_rand == "lambda.min":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights=n * lam_min * np.ones(p),
                                          randomizer_scale= np.sqrt(n) * 
                                          randomizer_scale * sigma_)
    elif tuning_rand == "lambda.1se":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights=n * lam_1se * np.ones(p),
                                          randomizer_scale= np.sqrt(n) * 
                                          randomizer_scale * sigma_)
    else:
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= lam_theory * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * 
                                          randomizer_scale * sigma_)
    signs = randomized_lasso.fit()
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())], np.bool)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)
    MLE_nreport = 0

    sys.stderr.write("active variables selected by cv LASSO  " + str(nactive_LASSO) + "\n")
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

    if nonzero.sum() > 0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)

        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        randomized_lasso_est = randomized_lasso.initial_soln
        randomized_rel_lasso_est = randomized_lasso._beta_full

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (
            np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

        partial_MLE_risk = (MLE_estimate - target_randomized).T.dot(MLE_estimate - target_randomized)
        partial_ind_risk = (ind_unbiased_estimator - target_randomized).T.dot(ind_unbiased_estimator - target_randomized)
        partial_randLasso_risk = (randomized_lasso_est[nonzero] - target_randomized).T.dot(randomized_lasso_est[nonzero] - target_randomized)
        partial_relrandLasso_risk = (randomized_rel_lasso_est[nonzero] - target_randomized).T.dot(randomized_rel_lasso_est[nonzero] - target_randomized)
    else:
        MLE_nreport = 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power = [0., 0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)
        partial_MLE_risk, partial_ind_risk, partial_randLasso_risk, partial_relrandLasso_risk = [0., 0., 0., 0.]

    risks = np.vstack((relative_risk(sel_MLE, beta, Sigma),
                       relative_risk(ind_est, beta, Sigma),
                       relative_risk(randomized_lasso_est, beta, Sigma),
                       relative_risk(randomized_rel_lasso_est, beta, Sigma),
                       relative_risk(rel_LASSO, beta, Sigma),
                       relative_risk(glm_LASSO, beta, Sigma)))

    partial_risks = np.vstack((partial_MLE_risk,
                               partial_ind_risk,
                               partial_randLasso_risk,
                               partial_relrandLasso_risk,
                               partial_relLasso_risk,
                               partial_Lasso_risk))

    naive_inf = np.vstack((cov_naive, length_naive, 0., nactive_LASSO, bias_naive, selective_naive_power, power_naive, power_naive_BH, fdr_naive_BH,
                           naive_discoveries.sum()))
    Lee_inf = np.vstack((cov_Lee, length_Lee, inf_entries, nactive_LASSO, bias_Lee, selective_Lee_power, power_Lee, power_Lee_BH, fdr_Lee_BH,
                         Lee_discoveries.sum()))
    Liu_inf = np.zeros((10, 1))
    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power, power_MLE, power_MLE_BH, fdr_MLE_BH,
                         MLE_discoveries.sum()))
    nreport = np.vstack((Lee_nreport, 0., MLE_nreport))
    return np.vstack((risks, naive_inf, Lee_inf, Liu_inf, MLE_inf, partial_risks, nreport))


def comparison_cvmetrics_full(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                              randomizer_scale=np.sqrt(0.25), full_dispersion=True,
                              tuning_nonrand="lambda.min", tuning_rand="lambda.1se"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    glm_LASSO_theory, glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory/float(n))
    if tuning_nonrand == "lambda.min":
        lam_LASSO = lam_min
        glm_LASSO = glm_LASSO_min
    elif tuning_nonrand == "lambda.1se":
        lam_LASSO = lam_1se
        glm_LASSO = glm_LASSO_1se
    else:
        lam_LASSO = lam_theory/float(n)
        glm_LASSO = glm_LASSO_theory

    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.asarray([(np.in1d(active_set_LASSO[z], true_set).sum() > 0) for z in range(nactive_LASSO)],
                                   np.bool)

    rel_LASSO = np.zeros(p)
    Lee_nreport = 0
    bias_Lee = 0.
    bias_naive = 0.

    if nactive_LASSO > 0:
        rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        Lee_target = beta[active_LASSO]
        Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_LASSO, sigma_, Type=1, alpha=0.1)

        if (Lee_pval.shape[0] == Lee_target.shape[0]):

            cov_Lee, selective_Lee_power = coverage(Lee_intervals, Lee_pval, Lee_target, beta[active_LASSO])
            inf_entries_bool = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            inf_entries = np.mean(inf_entries_bool)
            if inf_entries == 1.:
                length_Lee = 0.
            else:
                length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries_bool])
            power_Lee = ((active_LASSO_bool) * (
                np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum() / float((beta != 0).sum())
            Lee_discoveries = BHfilter(Lee_pval, q=0.1)
            power_Lee_BH = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_Lee_BH = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
            bias_Lee = np.mean(glm_LASSO[active_LASSO] - Lee_target)

            post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
            naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                         post_LASSO_OLS + 1.65 * naive_sd]).T
            naive_pval = 2 * ndist.cdf(np.abs(post_LASSO_OLS) / naive_sd)
            cov_naive, selective_naive_power = coverage(naive_intervals, naive_pval, Lee_target, beta[active_LASSO])
            length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
            power_naive = ((active_LASSO_bool) * (
                np.logical_or((0. < naive_intervals[:, 0]), (0. > naive_intervals[:, 1])))).sum() / float(
                (beta != 0).sum())
            naive_discoveries = BHfilter(naive_pval, q=0.1)
            power_naive_BH = (naive_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_naive_BH = (naive_discoveries * ~active_LASSO_bool).sum() / float(max(naive_discoveries.sum(), 1.))
            bias_naive = np.mean(rel_LASSO[active_LASSO] - Lee_target)

            partial_Lasso_risk = (glm_LASSO[active_LASSO] - Lee_target).T.dot(glm_LASSO[active_LASSO] - Lee_target)
            partial_relLasso_risk = (post_LASSO_OLS - Lee_target).T.dot(post_LASSO_OLS - Lee_target)
        else:
            Lee_nreport = 1
            cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, selective_Lee_power = [0., 0., 0., 0., 0., 0., 0.]
            cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH, selective_naive_power  = [0., 0., 0., 0., 0., 0.]
            naive_discoveries = np.zeros(1)
            Lee_discoveries = np.zeros(1)
            partial_Lasso_risk, partial_relLasso_risk = [0., 0.]

    elif nactive_LASSO == 0:
        Lee_nreport = 1
        cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, selective_Lee_power = [0., 0., 0., 0., 0., 0., 0.]
        cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH, selective_naive_power = [0., 0., 0., 0., 0., 0.]
        naive_discoveries = np.zeros(1)
        Lee_discoveries = np.zeros(1)
        partial_Lasso_risk, partial_relLasso_risk = [0., 0.]

    lasso_Liu = ROSI.gaussian(X, y, n * lam_LASSO)
    Lasso_soln_Liu = lasso_Liu.fit()
    active_set_Liu = np.nonzero(Lasso_soln_Liu != 0)[0]
    nactive_Liu = active_set_Liu.shape[0]
    active_Liu_bool = np.asarray([(np.in1d(active_set_Liu[a], true_set).sum() > 0) for a in range(nactive_Liu)], np.bool)
    Liu_nreport = 0

    if nactive_Liu > 0:
        Liu_target = beta[Lasso_soln_Liu != 0]
        df = lasso_Liu.summary(level=0.90, compute_intervals=True, dispersion=dispersion)
        Liu_lower, Liu_upper, Liu_pval = np.asarray(df['lower_confidence']), \
                                         np.asarray(df['upper_confidence']), \
                                         np.asarray(df['pval'])
        Liu_intervals = np.vstack((Liu_lower, Liu_upper)).T
        cov_Liu, selective_Liu_power = coverage(Liu_intervals, Liu_pval, Liu_target, beta[Lasso_soln_Liu != 0])
        length_Liu = np.mean(Liu_intervals[:, 1] - Liu_intervals[:, 0])
        power_Liu = ((active_Liu_bool) * (np.logical_or((0. < Liu_intervals[:, 0]),
                                                        (0. > Liu_intervals[:, 1])))).sum() / float((beta != 0).sum())
        Liu_discoveries = BHfilter(Liu_pval, q=0.1)
        power_Liu_BH = (Liu_discoveries * active_Liu_bool).sum() / float((beta != 0).sum())
        fdr_Liu_BH = (Liu_discoveries * ~active_Liu_bool).sum() / float(max(Liu_discoveries.sum(), 1.))

    else:
        Liu_nreport = 1
        cov_Liu, length_Liu, power_Liu, power_Liu_BH, fdr_Liu_BH, selective_Liu_power = [0., 0., 0., 0., 0., 0.]
        Liu_discoveries = np.zeros(1)

    if tuning_rand == "lambda.min":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_min * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    elif tuning_rand == "lambda.1se":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_1se * np.ones(p),
                                          randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)
    else:
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= lam_theory * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    signs = randomized_lasso.fit()
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())], np.bool)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)
    MLE_nreport = 0

    if nonzero.sum() > 0:
        target_randomized = beta[nonzero]
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = full_targets(randomized_lasso.loglike,
                                      randomized_lasso._W,
                                      nonzero,
                                      dispersion=dispersion)
        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        randomized_lasso_est = randomized_lasso.initial_soln
        randomized_rel_lasso_est = randomized_lasso._beta_full

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

        partial_MLE_risk = (MLE_estimate - target_randomized).T.dot(MLE_estimate - target_randomized)
        partial_ind_risk = (ind_unbiased_estimator - target_randomized).T.dot(ind_unbiased_estimator - target_randomized)
        partial_randLasso_risk = (randomized_lasso_est[nonzero] - target_randomized).T.dot(randomized_lasso_est[nonzero] - target_randomized)
        partial_relrandLasso_risk = (randomized_rel_lasso_est[nonzero] - target_randomized).T.dot(randomized_rel_lasso_est[nonzero] - target_randomized)
    else:
        MLE_nreport = 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power = [0., 0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)
        partial_MLE_risk, partial_ind_risk, partial_randLasso_risk, partial_relrandLasso_risk = [0., 0., 0., 0.]

    risks = np.vstack((relative_risk(sel_MLE, beta, Sigma),
                       relative_risk(ind_est, beta, Sigma),
                       relative_risk(randomized_lasso_est, beta, Sigma),
                       relative_risk(randomized_rel_lasso_est, beta, Sigma),
                       relative_risk(rel_LASSO, beta, Sigma),
                       relative_risk(glm_LASSO, beta, Sigma)))

    partial_risks = np.vstack((partial_MLE_risk,
                               partial_ind_risk,
                               partial_randLasso_risk,
                               partial_relrandLasso_risk,
                               partial_relLasso_risk,
                               partial_Lasso_risk))

    naive_inf = np.vstack((cov_naive, length_naive, 0., nactive_LASSO, bias_naive, selective_naive_power,
                           power_naive, power_naive_BH, fdr_naive_BH, naive_discoveries.sum()))
    Lee_inf = np.vstack((cov_Lee, length_Lee, inf_entries, nactive_LASSO, bias_Lee, selective_Lee_power,
                         power_Lee, power_Lee_BH, fdr_Lee_BH, Lee_discoveries.sum()))
    Liu_inf = np.vstack((cov_Liu, length_Liu, 0., nactive_Liu, bias_Lee, selective_Liu_power,
                         power_Liu, power_Liu_BH, fdr_Liu_BH, Liu_discoveries.sum()))
    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power,
                         power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum()))
    nreport = np.vstack((Lee_nreport, Liu_nreport, MLE_nreport))
    return np.vstack((risks, naive_inf, Lee_inf, Liu_inf, MLE_inf, partial_risks, nreport))

def comparison_cvmetrics_debiased(n=100, p=150, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                                  randomizer_scale=np.sqrt(0.25), full_dispersion=False,
                                  tuning_nonrand="lambda.min", tuning_rand="lambda.1se"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        _sigma_ = np.std(y)

    lam_theory = _sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    glm_LASSO_theory, glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory / float(n))

    if full_dispersion is False:
        dispersion = None
        active_min = (glm_LASSO_min != 0)
        if active_min.sum() > 0:
            sigma_ = np.sqrt(np.linalg.norm(y - X[:, active_min].dot(np.linalg.pinv(X[:, active_min]).dot(y))) ** 2
                             / (n - active_min.sum()))
        else:
            sigma_ = _sigma_
    print("estimated and true sigma", sigma, _sigma_, sigma_)

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
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.asarray([(np.in1d(active_set_LASSO[z], true_set).sum() > 0) for z in range(nactive_LASSO)],
                                   np.bool)

    rel_LASSO = np.zeros(p)
    Lee_nreport = 0.
    bias_naive = 0.

    if nactive_LASSO > 0:
        rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        Lee_target = beta[active_LASSO]
        post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
        naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                     post_LASSO_OLS + 1.65 * naive_sd]).T
        naive_pval = 2 * ndist.cdf(np.abs(post_LASSO_OLS) / naive_sd)
        cov_naive, selective_naive_power = coverage(naive_intervals, naive_pval, Lee_target, beta[active_LASSO])
        length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
        power_naive = ((active_LASSO_bool) * (
            np.logical_or((0. < naive_intervals[:, 0]), (0. > naive_intervals[:, 1])))).sum() / float(
            (beta != 0).sum())
        naive_discoveries = BHfilter(naive_pval, q=0.1)
        power_naive_BH = (naive_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
        fdr_naive_BH = (naive_discoveries * ~active_LASSO_bool).sum() / float(max(naive_discoveries.sum(), 1.))
        bias_naive = np.mean(rel_LASSO[active_LASSO] - Lee_target)

        partial_Lasso_risk = (glm_LASSO[active_LASSO] - Lee_target).T.dot(glm_LASSO[active_LASSO] - Lee_target)
        partial_relLasso_risk = (post_LASSO_OLS - Lee_target).T.dot(post_LASSO_OLS - Lee_target)

    elif nactive_LASSO == 0:
        Lee_nreport += 1
        cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH, selective_naive_power = [0., 0., 0., 0., 0., 0.]
        naive_discoveries = np.zeros(1)
        partial_Lasso_risk, partial_relLasso_risk = [0., 0.]

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
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())],
                                  np.bool)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)
    MLE_nreport = 0

    if nonzero.sum() > 0:
        target_randomized = beta[nonzero]
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = debiased_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          penalty=randomized_lasso.penalty,
                                          dispersion=dispersion)
        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        randomized_lasso_est = randomized_lasso.initial_soln
        randomized_rel_lasso_est = randomized_lasso._beta_full

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (
            np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

        partial_MLE_risk = (MLE_estimate - target_randomized).T.dot(MLE_estimate - target_randomized)
        partial_ind_risk = (ind_unbiased_estimator - target_randomized).T.dot(
            ind_unbiased_estimator - target_randomized)
        partial_randLasso_risk = (randomized_lasso_est[nonzero] - target_randomized).T.dot(
            randomized_lasso_est[nonzero] - target_randomized)
        partial_relrandLasso_risk = (randomized_rel_lasso_est[nonzero] - target_randomized).T.dot(
            randomized_rel_lasso_est[nonzero] - target_randomized)
    else:
        MLE_nreport = 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power = [0., 0., 0., 0., 0.,
                                                                                                   0., 0.]
        MLE_discoveries = np.zeros(1)
        partial_MLE_risk, partial_ind_risk, partial_randLasso_risk, partial_relrandLasso_risk = [0., 0., 0., 0.]

    risks = np.vstack((relative_risk(sel_MLE, beta, Sigma),
                       relative_risk(ind_est, beta, Sigma),
                       relative_risk(randomized_lasso_est, beta, Sigma),
                       relative_risk(randomized_rel_lasso_est, beta, Sigma),
                       relative_risk(rel_LASSO, beta, Sigma),
                       relative_risk(glm_LASSO, beta, Sigma)))

    partial_risks = np.vstack((partial_MLE_risk,
                               partial_ind_risk,
                               partial_randLasso_risk,
                               partial_relrandLasso_risk,
                               partial_relLasso_risk,
                               partial_Lasso_risk))

    naive_inf = np.vstack((cov_naive, length_naive, 0., nactive_LASSO, bias_naive, selective_naive_power,
                           power_naive, power_naive_BH, fdr_naive_BH, naive_discoveries.sum()))
    Lee_inf = np.zeros((10,1))
    Liu_inf = np.zeros((10,1))
    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power,
                         power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum()))
    nreport = np.vstack((Lee_nreport, 0., MLE_nreport))
    return np.vstack((risks, naive_inf, Lee_inf, Liu_inf, MLE_inf, partial_risks, nreport))


def compare_sampler_MLE(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20, target= "selected",
                        randomizer_scale=np.sqrt(0.50), full_dispersion=True, tuning_rand="lambda.1se"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    _, _, _, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory / float(n))

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
    elif tuning_rand == "lambda.theory":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights=lam_theory * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    else:
        raise ValueError('lambda choice not specified correctly')

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())],
                                  np.bool)
    nreport = 0.

    if nonzero.sum() > 0:
        if target == "full":
            target_randomized = beta[nonzero]
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = full_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)
        elif target == "selected":
            target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(randomized_lasso.loglike,
                                              randomized_lasso._W,
                                              nonzero,
                                              dispersion=dispersion)
        else:
            raise ValueError('not a valid specification of target')
        toc = time.time()
        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        tic = time.time()
        time_MLE = tic - toc

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (
            np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

        toc = time.time()
        _, sampler_pval, sampler_intervals = randomized_lasso.summary(observed_target,
                                                                      cov_target,
                                                                      cov_target_score,
                                                                      alternatives,
                                                                      level=0.9, compute_intervals=True, ndraw=100000)
        tic = time.time()
        time_sampler = tic - toc

        cov_sampler, selective_sampler_power = coverage(sampler_intervals, sampler_pval, target_randomized, beta[nonzero])
        length_sampler = np.mean(sampler_intervals[:, 1] - sampler_intervals[:, 0])
        power_sampler = ((active_rand_bool) * (np.logical_or((0. < sampler_intervals[:, 0]),
                                                             (0. > sampler_intervals[:, 1])))).sum() / float((beta != 0).sum())
        sampler_discoveries = BHfilter(sampler_pval, q=0.1)
        power_sampler_BH = (sampler_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_sampler_BH = (sampler_discoveries * ~active_rand_bool).sum() / float(max(sampler_discoveries.sum(), 1.))
        bias_randLASSO = np.mean(randomized_lasso.initial_soln[nonzero] - target_randomized)

    else:
        nreport += 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power, time_MLE = [0., 0., 0., 0., 0., 0., 0., 0.]
        cov_sampler, length_sampler, power_sampler, power_sampler_BH, fdr_sampler_BH, bias_randLASSO, selective_sampler_power, time_sampler = [0., 0., 0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)
        sampler_discoveries = np.zeros(1)

    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power, time_MLE,
                         power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum()))

    sampler_inf = np.vstack((cov_sampler, length_sampler, 0., nonzero.sum(), bias_randLASSO, selective_sampler_power, time_sampler,
                             power_sampler, power_sampler_BH, fdr_sampler_BH, sampler_discoveries.sum()))

    return np.vstack((MLE_inf, sampler_inf, nreport))









