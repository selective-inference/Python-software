import numpy as np, sys

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from selection.algorithms.lasso import lasso
from scipy.stats import norm as ndist

def BHfilter(pval, q=0.2):
    robjects.r.assign('pval', pval)
    robjects.r.assign('q', q)
    robjects.r('Pval = p.adjust(pval, method="BH")')
    robjects.r('S = which((Pval < q)) - 1')
    S = robjects.r('S')
    ind = np.zeros(pval.shape[0], np.bool)
    ind[np.asarray(S, np.int)] = 1
    return ind

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
               #print(inf$ci)
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
                glmnet_LASSO = function(X,y,lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    return estimate

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    library(bestsubset)
    sim_xy = bestsubset::sim.xy
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

    return X, y, X_val, y_val, Sigma, beta, sigma

def tuned_lasso(X, y, X_val,y_val):
    robjects.r('''
        tuned_lasso_estimator = function(X,Y,X.val,Y.val){
        Y = as.matrix(Y)
        X = as.matrix(X)
        Y.val = as.vector(Y.val)
        X.val = as.matrix(X.val)
        rel.LASSO = lasso(X,Y,intercept=TRUE, nrelax=10, nlam=50, standardize=TRUE)
        LASSO = lasso(X,Y,intercept=TRUE,nlam=50, standardize=TRUE)
        beta.hat.rellasso = as.matrix(coef(rel.LASSO))
        beta.hat.lasso = as.matrix(coef(LASSO))
        min.lam = min(rel.LASSO$lambda)
        max.lam = max(rel.LASSO$lambda)

        lam.seq = exp(seq(log(max.lam),log(min.lam),length=rel.LASSO$nlambda))
  
        muhat.val.rellasso = as.matrix(predict(rel.LASSO, X.val))
        muhat.val.lasso = as.matrix(predict(LASSO, X.val))
        err.val.rellasso = colMeans((muhat.val.rellasso - Y.val)^2)
        err.val.lasso = colMeans((muhat.val.lasso - Y.val)^2)

        opt_lam = ceiling(which.min(err.val.rellasso)/10)
        lambda.tuned.rellasso = lam.seq[opt_lam]
        lambda.tuned.lasso = lam.seq[which.min(err.val.lasso)]
        fit = glmnet(X, Y, standardize=TRUE, intercept=TRUE)
        estimate.tuned = coef(fit, s=lambda.tuned.lasso, exact=TRUE, x=X, y=Y)[-1]
        beta.hat.lasso = (beta.hat.lasso[,which.min(err.val.lasso)])[-1]
        return(list(beta.hat.rellasso = (beta.hat.rellasso[,which.min(err.val.rellasso)])[-1],
        beta.hat.lasso = beta.hat.lasso,
        lambda.tuned.rellasso = lambda.tuned.rellasso, lambda.tuned.lasso= lambda.tuned.lasso,
        lambda.seq = lam.seq))
        }''')

    r_lasso = robjects.globalenv['tuned_lasso_estimator']

    n, p = X.shape
    nval, _ = X_val.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_X_val = robjects.r.matrix(X_val, nrow=nval, ncol=p)
    r_y_val = robjects.r.matrix(y_val, nrow=nval, ncol=1)

    tuned_est = r_lasso(r_X, r_y, r_X_val, r_y_val)
    estimator_rellasso = np.array(tuned_est.rx2('beta.hat.rellasso'))
    estimator_lasso = np.array(tuned_est.rx2('beta.hat.lasso'))
    lam_tuned_rellasso = np.asscalar(np.array(tuned_est.rx2('lambda.tuned.rellasso')))
    lam_tuned_lasso = np.asscalar(np.array(tuned_est.rx2('lambda.tuned.lasso')))
    lam_seq = np.array(tuned_est.rx2('lambda.seq'))
    return estimator_rellasso, estimator_lasso, lam_tuned_rellasso, lam_tuned_lasso, lam_seq

def relative_risk(est, truth, Sigma):

    return (est-truth).T.dot(Sigma).dot(est-truth)/truth.T.dot(Sigma).dot(truth)

def coverage(intervals, pval, truth):
    if (truth!=0).sum()!=0:
        avg_power = np.mean(pval[truth != 0])
    else:
        avg_power = 0.
    return np.mean((truth > intervals[:, 0])*(truth < intervals[:, 1])), avg_power


def comparison_risk_inference_selected(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=2, snr=0.20,
                                       randomizer_scale=np.sqrt(0.25), target = "selected",
                                       tuning = "selective_MLE", full_dispersion = True):

    while True:
        X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho,
                                                        s=s, beta_type=beta_type, snr=snr)
        true_mean = X.dot(beta)
        rel_LASSO, est_LASSO, lam_tuned_rellasso, lam_tuned_lasso, lam_seq = tuned_lasso(X, y, X_val, y_val)
        active_nonrand = (est_LASSO != 0)
        nactive_nonrand = active_nonrand.sum()

        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        X_val -= X_val.mean(0)[None, :]
        X_val /= (X_val.std(0)[None, :] * np.sqrt(n / (n - 1.)))

        y = y - y.mean()
        y_val = y_val - y_val.mean()

        if full_dispersion:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        else:
            dispersion = None

        sigma_ = np.sqrt(dispersion)
        print("estimated and true sigma", sigma, sigma_)

        glm_LASSO = glmnet_lasso(X, y, lam_tuned_lasso)
        active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()

        tune_num = 50
        lam_seq = sigma_ * np.linspace(0.25, 2.75, num=tune_num) * \
                  np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        err = np.zeros(tune_num)
        for k in range(tune_num):
            W = lam_seq[k] * np.ones(p)
            conv = lasso.gaussian(X,
                                  y,
                                  W,
                                  randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
            signs = conv.fit()
            nonzero = signs != 0
            if tuning == "selective_MLE":
                estimate, _, _, _, _, _ = conv.selective_MLE(target=target, dispersion=dispersion)
                full_estimate = np.zeros(p)
                full_estimate[nonzero] = estimate
                err[k] = np.mean((y_val - X_val.dot(full_estimate)) ** 2.)
            elif tuning == "randomized_LASSO":
                err[k] = np.mean((y_val - X_val.dot(conv.initial_soln)) ** 2.)


        lam = lam_seq[np.argmin(err)]
        sys.stderr.write("lambda from randomized LASSO " + str(lam) + "\n")
        # print(lam_tuned_lasso * n, lam, lam_seq)

        randomized_lasso = highdim.gaussian(X,
                                            y,
                                            lam * np.ones(p),
                                            randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        signs = randomized_lasso.fit()
        nonzero = signs != 0

        sys.stderr.write("active variables selected by tuned LASSO " + str(nactive_nonrand) + "\n")
        sys.stderr.write("active variables selected by LASSO in python " + str(nactive_LASSO) + "\n")
        sys.stderr.write("recall glmnet at tuned lambda " + str((glm_LASSO != 0).sum()) + "\n")
        sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

        if nactive_LASSO>0 and nonzero.sum()>0 and nactive_nonrand>0:
            beta_target_rand = np.linalg.pinv(X[:, nonzero]).dot(true_mean)
            beta_target_nonrand_py = np.linalg.pinv(X[:, active_LASSO]).dot(true_mean)
            beta_target_nonrand = np.linalg.pinv(X[:, active_nonrand]).dot(true_mean)

            Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_tuned_lasso, sigma_, Type=0, alpha=0.1)

            if (Lee_pval.shape[0] == beta_target_nonrand_py.shape[0]):
                sel_MLE = np.zeros(p)
                estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(
                    target=target,
                    dispersion=dispersion)
                sel_MLE[nonzero] = estimate
                ind_estimator = np.zeros(p)
                ind_estimator[nonzero] = ind_unbiased_estimator

                post_LASSO_OLS = np.linalg.pinv(X[:, active_nonrand]).dot(y)
                unad_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_nonrand].T.dot(X[:, active_nonrand])))))
                unad_intervals = np.vstack([post_LASSO_OLS - 1.65 * unad_sd,
                                            post_LASSO_OLS + 1.65 * unad_sd]).T
                unad_pval = ndist.cdf(post_LASSO_OLS / unad_sd)

                true_signals = np.zeros(p, np.bool)
                true_signals[beta != 0] = 1
                true_set = np.asarray([u for u in range(p) if true_signals[u]])
                active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
                active_set_nonrand = np.asarray([q for q in range(p) if active_nonrand[q]])
                active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])

                active_rand_bool = np.zeros(nonzero.sum(), np.bool)
                for x in range(nonzero.sum()):
                    active_rand_bool[x] = (np.in1d(active_set_rand[x], true_set).sum() > 0)
                active_nonrand_bool = np.zeros(nactive_nonrand, np.bool)
                for w in range(nactive_nonrand):
                    active_nonrand_bool[w] = (np.in1d(active_set_nonrand[w], true_set).sum() > 0)
                active_LASSO_bool = np.zeros(nactive_LASSO, np.bool)
                for z in range(nactive_LASSO):
                    active_LASSO_bool[z] = (np.in1d(active_set_LASSO[z], true_set).sum() > 0)

                cov_sel, _ = coverage(sel_intervals, sel_pval, beta_target_rand)
                # print("check shapes", Lee_pval.shape, beta_target_nonrand_py.shape, Lee_pval)
                cov_Lee, _ = coverage(Lee_intervals, Lee_pval, beta_target_nonrand_py)
                cov_unad, _ = coverage(unad_intervals, unad_pval, beta_target_nonrand)

                power_sel = (
                (active_rand_bool) * (np.logical_or((0. < sel_intervals[:, 0]), (0. > sel_intervals[:, 1])))).sum()
                power_Lee = (
                (active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum()
                power_unad = (
                (active_nonrand_bool) * (np.logical_or((0. < unad_intervals[:, 0]), (0. > unad_intervals[:, 1])))).sum()

                sel_discoveries = BHfilter(sel_pval, q=0.1)
                Lee_discoveries = BHfilter(Lee_pval, q=0.1)
                unad_discoveries = BHfilter(unad_pval, q=0.1)

                power_sel_dis = (sel_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
                power_Lee_dis = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
                power_unad_dis = (unad_discoveries * active_nonrand_bool).sum() / float((beta != 0).sum())

                fdr_sel_dis = (sel_discoveries * ~active_rand_bool).sum() / float(max(sel_discoveries.sum(), 1.))
                fdr_Lee_dis = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
                fdr_unad_dis = (unad_discoveries * ~active_nonrand_bool).sum() / float(max(unad_discoveries.sum(), 1.))
                break

    if True:
        return np.vstack((relative_risk(sel_MLE, beta, Sigma),
                          relative_risk(ind_estimator, beta, Sigma),
                          relative_risk(randomized_lasso.initial_soln , beta, Sigma),
                          relative_risk(randomized_lasso._beta_full, beta, Sigma),
                          relative_risk(rel_LASSO, beta, Sigma),
                          relative_risk(est_LASSO, beta, Sigma),
                          cov_sel,
                          cov_Lee,
                          cov_unad,
                          np.mean(sel_intervals[:, 1] - sel_intervals[:, 0]),
                          np.mean(Lee_intervals[:, 1] - Lee_intervals[:, 0]),
                          np.mean(unad_intervals[:, 1] - unad_intervals[:, 0]),
                          power_sel/float((beta != 0).sum()),
                          power_Lee/float((beta != 0).sum()),
                          power_unad/float((beta != 0).sum()),
                          power_sel_dis,
                          power_Lee_dis,
                          power_unad_dis,
                          fdr_sel_dis,
                          fdr_Lee_dis,
                          fdr_unad_dis,
                          nonzero.sum(),
                          nactive_LASSO,
                          nactive_nonrand,
                          sel_discoveries.sum(),
                          Lee_discoveries.sum(),
                          unad_discoveries.sum()))

def comparison_risk_inference_full(n=200, p=500, nval=200, rho=0.35, s=5, beta_type=2,
                                   snr=0.2, randomizer_scale=0.5, target = "full",
                                   tuning = "selective_MLE", full_dispersion = True):

    while True:
        X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho,
                                                        s=s, beta_type=beta_type, snr=snr)
        rel_LASSO, est_LASSO, lam_tuned_rellasso, lam_tuned_lasso, lam_seq = tuned_lasso(X, y, X_val, y_val)
        active_nonrand = (est_LASSO != 0)
        nactive_nonrand = active_nonrand.sum()

        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n/(n-1.)))
        X_val -= X_val.mean(0)[None, :]
        X_val /= (X_val.std(0)[None, :] * np.sqrt(n/(n-1.)))

        y = y - y.mean()
        y_val = y_val - y_val.mean()

        if full_dispersion:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)

        print("estimated and true sigma", sigma, sigma_)

        glm_LASSO = glmnet_lasso(X, y, lam_tuned_lasso)
        active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()

        tune_num = 50
        rand_tune_num = 10
        rand_scale_seq = np.linspace(0.05, 0.5, num = rand_tune_num)
        lam_seq = sigma_ * np.linspace(0.25, 2.75, num=tune_num) * \
                  np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        err = np.zeros((rand_tune_num, tune_num))
        for l in range(rand_tune_num):
            randomizer_scale = rand_scale_seq[l]
            for k in range(tune_num):
                W = lam_seq[k] * np.ones(p)
                conv = highdim.gaussian(X,
                                        y,
                                        W,
                                        randomizer_scale=np.sqrt(n) *
                                                         randomizer_scale * sigma_)
                signs = conv.fit()
                nonzero = signs != 0
                if tuning == "selective_MLE":
                    estimate, _, _, _, _, _ = conv.selective_MLE(target=target, dispersion=dispersion)
                    full_estimate = np.zeros(p)
                    full_estimate[nonzero] = estimate
                    err[l, k] = np.mean((y_val - X_val.dot(full_estimate)) ** 2.)
                elif tuning == "randomized_LASSO":
                    err[l, k] = np.mean((y_val - X_val.dot(conv.initial_soln)) ** 2.)

        arg_min = np.argwhere(err == np.min(err))
        lam = lam_seq[arg_min[0, 1]]
        randomizer_scale = rand_scale_seq[arg_min[0, 0]]
        #lam = lam_seq[np.argmin(err)]
        sys.stderr.write("lambda from randomized LASSO " + str(lam) + "\n")
        sys.stderr.write("tuned randomized scale " + str(randomizer_scale) + "\n")
        #print(lam_tuned_lasso * n, lam, lam_seq)

        randomized_lasso = highdim.gaussian(X,
                                            y,
                                            lam * np.ones(p),
                                            randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        signs = randomized_lasso.fit()
        nonzero = signs != 0
        sys.stderr.write("active variables selected by tuned LASSO " + str(nactive_nonrand) + "\n")
        sys.stderr.write("active variables selected by LASSO in python " + str(nactive_LASSO) + "\n")
        sys.stderr.write("recall glmnet at tuned lambda " + str((glm_LASSO != 0).sum()) + "\n")
        sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

        if nonzero.sum()>0 and nactive_nonrand>0 and nonzero.sum()<50:
            beta_target_rand = beta[nonzero]
            beta_target_nonrand_py = beta[active_LASSO]
            beta_target_nonrand = beta[active_nonrand]

            Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_tuned_lasso, sigma_, Type=1, alpha=0.1)

            if (Lee_pval.shape[0] == beta_target_nonrand_py.shape[0]):
                sel_MLE = np.zeros(p)
                estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(
                    target=target,
                    dispersion=dispersion)
                sel_MLE[nonzero] = estimate
                ind_estimator = np.zeros(p)
                ind_estimator[nonzero] = ind_unbiased_estimator

                if Lee_pval.shape[0] != beta_target_nonrand_py.shape[0]:
                    break

                post_LASSO_OLS = np.linalg.pinv(X[:, active_nonrand]).dot(y)
                unad_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_nonrand].T.dot(X[:, active_nonrand])))))

                unad_intervals = np.vstack([post_LASSO_OLS - 1.65 * unad_sd,
                                            post_LASSO_OLS + 1.65 * unad_sd]).T
                unad_pval = ndist.cdf(post_LASSO_OLS / unad_sd)

                true_signals = np.zeros(p, np.bool)
                true_signals[beta != 0] = 1
                true_set = np.asarray([u for u in range(p) if true_signals[u]])
                active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
                active_set_nonrand = np.asarray([q for q in range(p) if active_nonrand[q]])
                active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])

                active_rand_bool = np.zeros(nonzero.sum(), np.bool)
                for x in range(nonzero.sum()):
                    active_rand_bool[x] = (np.in1d(active_set_rand[x], true_set).sum() > 0)
                active_nonrand_bool = np.zeros(nactive_nonrand, np.bool)
                for w in range(nactive_nonrand):
                    active_nonrand_bool[w] = (np.in1d(active_set_nonrand[w], true_set).sum() > 0)
                active_LASSO_bool = np.zeros(nactive_LASSO, np.bool)
                for z in range(nactive_LASSO):
                    active_LASSO_bool[z] = (np.in1d(active_set_LASSO[z], true_set).sum() > 0)

                cov_sel, _ = coverage(sel_intervals, sel_pval, beta_target_rand)
                cov_Lee, _ = coverage(Lee_intervals, Lee_pval, beta_target_nonrand_py)
                cov_unad, _ = coverage(unad_intervals, unad_pval, beta_target_nonrand)

                power_sel = ((active_rand_bool) * (np.logical_or((0. < sel_intervals[:, 0]),
                                                                 (0. > sel_intervals[:, 1])))).sum()
                power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]),
                                                                  (0. > Lee_intervals[:, 1])))).sum()
                power_unad = ((active_nonrand_bool) * (np.logical_or((0. < unad_intervals[:, 0]),
                                                                     (0. > unad_intervals[:, 1])))).sum()

                sel_discoveries = BHfilter(sel_pval, q=0.1)
                Lee_discoveries = BHfilter(Lee_pval, q=0.1)
                unad_discoveries = BHfilter(unad_pval, q=0.1)

                power_sel_dis = (sel_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
                power_Lee_dis = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
                power_unad_dis = (unad_discoveries * active_nonrand_bool).sum() / float((beta != 0).sum())

                fdr_sel_dis = (sel_discoveries * ~active_rand_bool).sum() / float(max(sel_discoveries.sum(), 1.))
                fdr_Lee_dis = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
                fdr_unad_dis = (unad_discoveries * ~active_nonrand_bool).sum() / float(max(unad_discoveries.sum(), 1.))
                break

    if True:
        return np.vstack((relative_risk(sel_MLE, beta, Sigma),
                          relative_risk(ind_estimator, beta, Sigma),
                          relative_risk(randomized_lasso.initial_soln , beta, Sigma),
                          relative_risk(randomized_lasso._beta_full, beta, Sigma),
                          relative_risk(rel_LASSO, beta, Sigma),
                          relative_risk(est_LASSO, beta, Sigma),
                          cov_sel,
                          cov_Lee,
                          cov_unad,
                          np.mean(sel_intervals[:, 1] - sel_intervals[:, 0]),
                          np.mean(Lee_intervals[:, 1] - Lee_intervals[:, 0]),
                          np.mean(unad_intervals[:, 1] - unad_intervals[:, 0]),
                          power_sel/float((beta != 0).sum()),
                          power_Lee/float((beta != 0).sum()),
                          power_unad/float((beta != 0).sum()),
                          power_sel_dis,
                          power_Lee_dis,
                          power_unad_dis,
                          fdr_sel_dis,
                          fdr_Lee_dis,
                          fdr_unad_dis,
                          nonzero.sum(),
                          nactive_LASSO,
                          nactive_nonrand,
                          sel_discoveries.sum(),
                          Lee_discoveries.sum(),
                          unad_discoveries.sum()))


if __name__ == "__main__":

    ndraw = 50
    output_overall = np.zeros(27)

    target = "selected"
    tuning = "selective_MLE"
    n, p, rho, s, beta_type, snr = 500, 100, 0.70, 5, 1, 0.10
    #nval = 100

    if target == "selected":
        for i in range(ndraw):
            output = comparison_risk_inference_selected(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                        randomizer_scale=np.sqrt(0.25), target=target, tuning= tuning,
                                                        full_dispersion=True)
            output_overall += np.squeeze(output)

            sys.stderr.write("overall selMLE risk " + str(output_overall[0] / float(i + 1)) + "\n")
            sys.stderr.write("overall indep est risk " + str(output_overall[1] / float(i + 1)) + "\n")
            sys.stderr.write("overall randomized LASSO est risk " + str(output_overall[2] / float(i + 1)) + "\n")
            sys.stderr.write(
                "overall relaxed rand LASSO est risk " + str(output_overall[3] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall relLASSO risk " + str(output_overall[4] / float(i + 1)) + "\n")
            sys.stderr.write("overall LASSO risk " + str(output_overall[5] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective coverage " + str(output_overall[6] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee coverage " + str(output_overall[7] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad coverage " + str(output_overall[8] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective length " + str(output_overall[9] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee length " + str(output_overall[10] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad length " + str(output_overall[11] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective power " + str(output_overall[12] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee power " + str(output_overall[13] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad power " + str(output_overall[14] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective fdr " + str(output_overall[18] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee fdr " + str(output_overall[19] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad fdr " + str(output_overall[20] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective power post BH " + str(output_overall[15] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee power post BH  " + str(output_overall[16] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad power post BH " + str(output_overall[17] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("average selective nactive " + str(output_overall[21] / float(i + 1)) + "\n")
            sys.stderr.write("average Lee nactive  " + str(output_overall[22] / float(i + 1)) + "\n")
            sys.stderr.write("average tuned LASSO nactive " + str(output_overall[23] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("average selective discoveries " + str(output_overall[24] / float(i + 1)) + "\n")
            sys.stderr.write("average Lee discoveries " + str(output_overall[25] / float(i + 1)) + "\n")
            sys.stderr.write("average tuned LASSO discoveries " + str(output_overall[26] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("iteration completed " + str(i + 1) + "\n")

    elif target == "full":
        if n > p:
            full_dispersion = True
        else:
            full_dispersion = False
        for i in range(ndraw):
            output = comparison_risk_inference_full(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                    randomizer_scale=np.sqrt(0.25), target=target, tuning= tuning,
                                                    full_dispersion=full_dispersion)
            output_overall += np.squeeze(output)

            sys.stderr.write("overall selMLE risk " + str(output_overall[0] / float(i + 1)) + "\n")
            sys.stderr.write("overall indep est risk " + str(output_overall[1] / float(i + 1)) + "\n")
            sys.stderr.write("overall randomized LASSO est risk " + str(output_overall[2] / float(i + 1)) + "\n")
            sys.stderr.write(
                "overall relaxed rand LASSO est risk " + str(output_overall[3] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall relLASSO risk " + str(output_overall[4] / float(i + 1)) + "\n")
            sys.stderr.write("overall LASSO risk " + str(output_overall[5] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective coverage " + str(output_overall[6] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee coverage " + str(output_overall[7] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad coverage " + str(output_overall[8] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective length " + str(output_overall[9] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee length " + str(output_overall[10] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad length " + str(output_overall[11] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective power " + str(output_overall[12] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee power " + str(output_overall[13] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad power " + str(output_overall[14] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective fdr " + str(output_overall[18] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee fdr " + str(output_overall[19] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad fdr " + str(output_overall[20] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selective power post BH " + str(output_overall[15] / float(i + 1)) + "\n")
            sys.stderr.write("overall Lee power post BH  " + str(output_overall[16] / float(i + 1)) + "\n")
            sys.stderr.write("overall unad power post BH " + str(output_overall[17] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("average selective nactive " + str(output_overall[21] / float(i + 1)) + "\n")
            sys.stderr.write("average Lee nactive  " + str(output_overall[22] / float(i + 1)) + "\n")
            sys.stderr.write("average tuned LASSO nactive " + str(output_overall[23] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("average selective discoveries " + str(output_overall[24] / float(i + 1)) + "\n")
            sys.stderr.write("average Lee discoveries " + str(output_overall[25] / float(i + 1)) + "\n")
            sys.stderr.write("average tuned LASSO discoveries " + str(output_overall[26] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("iteration completed " + str(i + 1) + "\n")
