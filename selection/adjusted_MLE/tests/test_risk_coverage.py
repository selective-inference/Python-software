import numpy as np, sys
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim

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
        rel.LASSO = lasso(X,Y,intercept=FALSE, nrelax=10, nlam=50)
        LASSO = lasso(X,Y,intercept=FALSE,nlam=50)
        beta.hat.rellasso = as.matrix(coef(rel.LASSO))
        beta.hat.lasso = as.matrix(coef(LASSO))
        min.lam = min(rel.LASSO$lambda)
        max.lam = max(rel.LASSO$lambda)
        lam.seq = exp(seq(log(max.lam),log(min.lam),length=rel.LASSO$nlambda))
        muhat.val.rellasso = as.matrix(predict(rel.LASSO, X.val))
        muhat.val.lasso = as.matrix(predict(LASSO, X.val))
        err.val.rellasso = colMeans((muhat.val.rellasso - Y.val)^2)
        err.val.lasso = colMeans((muhat.val.lasso - Y.val)^2)
        #print(err.val.rellasso)
        opt_lam = ceiling(which.min(err.val.rellasso)/10)
        lambda.tuned = lam.seq[opt_lam]
        return(list(beta.hat.rellasso = beta.hat.rellasso[,which.min(err.val.rellasso)],
        beta.hat.lasso = beta.hat.lasso[,which.min(err.val.lasso)],
        lambda.tuned = lambda.tuned, lambda.seq = lam.seq))
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
    lam_tuned = np.array(tuned_est.rx2('lambda.tuned'))
    lam_seq = np.array(tuned_est.rx2('lambda.seq'))
    return estimator_rellasso, estimator_lasso, lam_tuned, lam_seq

def relative_risk(est, truth, Sigma):

    return (est-truth).T.dot(Sigma).dot(est-truth)/truth.T.dot(Sigma).dot(truth)

def comparison_risk_inference(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=2, snr=0.2,
                              randomizer_scale=np.sqrt(0.25), target = "selected",
                              full_dispersion = True):

    X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho,
                                                    s=s, beta_type=beta_type, snr=snr)
    rel_LASSO, est_LASSO, lam_tuned, lam_seq = tuned_lasso(X, y, X_val, y_val)
    active_nonrand = (rel_LASSO != 0)
    nactive_nonrand = active_nonrand.sum()
    true_mean = X.dot(beta)

    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_val -= X_val.mean(0)[None, :]
    X_val /= (X_val.std(0)[None, :] * np.sqrt(nval))

    sigma_ = np.std(y)
    print("naive estimate of sigma_", sigma_)

    _y = y
    y = y - y.mean()
    y_val = y_val - y_val.mean()

    dispersion = None
    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)

    const = highdim.gaussian
    lam_seq = sigma_* np.linspace(0.75, 2.75, num=100) * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    err = np.zeros(100)
    for k in range(100):
        W = lam_seq[k]
        conv = const(X,
                     y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)
        signs = conv.fit()
        nonzero = signs != 0
        estimate, _, _, _, _ = conv.selective_MLE(target=target, dispersion=dispersion)

        full_estimate = np.zeros(p)
        full_estimate[nonzero] = estimate
        err[k] = np.mean((y_val - X_val.dot(full_estimate)) ** 2.)

    lam = lam_seq[np.argmin(err)]
    sys.stderr.write("lambda from tuned relaxed LASSO" + str(sigma_*lam_tuned) + "\n")
    sys.stderr.write("lambda from randomized LASSO" + str(lam) + "\n")

    randomized_lasso = const(X,
                             y,
                             lam,
                             randomizer_scale=randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0

    print("nonzero", nonzero.sum())
    sel_MLE = np.zeros(p)
    estimate, _, _, pval, intervals = randomized_lasso.selective_MLE(target=target, dispersion=dispersion)
    sel_MLE[nonzero] = estimate / np.sqrt(n)

    sys.stderr.write("overall_selrisk" + str(relative_risk(rel_LASSO, beta, Sigma)) + "\n")
    sys.stderr.write("overall_relLASSOrisk" + str(relative_risk(sel_MLE, beta, Sigma)) + "\n")


comparison_risk_inference(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=2, snr=0.2,
                          randomizer_scale=np.sqrt(0.25), target = "selected", full_dispersion = True)

