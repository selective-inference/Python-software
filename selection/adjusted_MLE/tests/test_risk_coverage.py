import numpy as np, sys
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import lasso, highdim

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                glmnet_LASSO = function(X,y,lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE)
                estimate = coef(fit, s=lam)[-1]
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
        #print(paste("max and min values of lambda", max.lam, min.lam))

        lam.seq = exp(seq(log(max.lam),log(min.lam),length=rel.LASSO$nlambda))
        muhat.val.rellasso = as.matrix(predict(rel.LASSO, X.val))
        muhat.val.lasso = as.matrix(predict(LASSO, X.val))
        err.val.rellasso = colMeans((muhat.val.rellasso - Y.val)^2)
        err.val.lasso = colMeans((muhat.val.lasso - Y.val)^2)

        opt_lam = ceiling(which.min(err.val.rellasso)/10)
        lambda.tuned.rellasso = lam.seq[opt_lam]
        lambda.tuned.lasso = lam.seq[which.min(err.val.lasso)]

        fit = glmnet(X, Y, standardize=TRUE, intercept=TRUE)
        estimate.tuned = coef(fit, s=lambda.tuned.lasso)[-1]

        #print(paste("compare estimates", max(abs(estimate.tuned-(beta.hat.lasso[,which.min(err.val.lasso)])[-1])),
        length(which(estimate.tuned!=0)), length(which((beta.hat.lasso[,which.min(err.val.lasso)])[-1]!=0))))

        return(list(beta.hat.rellasso = (beta.hat.rellasso[,which.min(err.val.rellasso)])[-1],
        beta.hat.lasso = (beta.hat.lasso[,which.min(err.val.lasso)])[-1],
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
    lam_tuned_rellasso = np.array(tuned_est.rx2('lambda.tuned.rellasso'))
    lam_tuned_lasso = np.array(tuned_est.rx2('lambda.tuned.lasso'))
    lam_seq = np.array(tuned_est.rx2('lambda.seq'))
    return estimator_rellasso, estimator_lasso, lam_tuned_rellasso, lam_tuned_lasso, lam_seq

def relative_risk(est, truth, Sigma):

    return (est-truth).T.dot(Sigma).dot(est-truth)/truth.T.dot(Sigma).dot(truth)

def comparison_risk_inference(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=2, snr=0.2,
                              randomizer_scale=np.sqrt(0.25), target = "selected",
                              full_dispersion = True):

    X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho,
                                                    s=s, beta_type=beta_type, snr=snr)
    rel_LASSO, est_LASSO, lam_tuned_rellasso, lam_tuned_lasso, lam_seq = tuned_lasso(X, y, X_val, y_val)
    active_nonrand = (est_LASSO != 0)
    nactive_nonrand = active_nonrand.sum()
    true_mean = X.dot(beta)

    _X = X
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_val -= X_val.mean(0)[None, :]
    X_val /= (X_val.std(0)[None, :] * np.sqrt(nval))

    _y = y
    y = y - y.mean()
    y_val = y_val - y_val.mean()

    dispersion = None
    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)

    sigma_ = np.std(y)
    LASSO_py = lasso.gaussian(X, y, np.asscalar((sigma**2.)*lam_tuned_lasso))
    soln = LASSO_py.fit()
    #print("compare solns", soln, est_LASSO)
    active_LASSO = (soln != 0)
    nactive_LASSO = active_LASSO.sum()

    # LASSO_rand0 = highdim.gaussian(X,
    #                                y,
    #                                np.asscalar((sigma_**2)*lam_tuned_lasso),
    #                                randomizer_scale=0.00000001)
    # signs_rand0 = LASSO_rand0.fit()

    #glm_LASSO = glmnet_lasso(X, y, np.asscalar(lam_tuned_lasso))

    const = highdim.gaussian
    lam_seq = sigma_* np.linspace(0.25, 2.75, num=100) * \
              np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    err = np.zeros(100)
    for k in range(100):
        W = lam_seq[k]
        conv = const(X,
                     y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)
        signs = conv.fit()
        nonzero = signs != 0
        estimate, _, _, _, _, _ = conv.selective_MLE(target=target, dispersion=dispersion)

        full_estimate = np.zeros(p)
        full_estimate[nonzero] = estimate
        err[k] = np.mean((y_val - X_val.dot(full_estimate)) ** 2.)

    lam = lam_seq[np.argmin(err)]
    sys.stderr.write("lambda from tuned relaxed LASSO " + str((sigma_**2)*lam_tuned_lasso) + "\n")
    sys.stderr.write("lambda from randomized LASSO " + str(lam) + "\n")

    randomized_lasso = const(X,
                             y,
                             lam,
                             randomizer_scale=randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    sys.stderr.write("active variables selected by tuned LASSO " + str(nactive_nonrand) + "\n")
    sys.stderr.write("active variables selected by LASSO in python " + str(nactive_LASSO)+ "\n")
    #sys.stderr.write("recall glmnet at tuned lambda " + str((glm_LASSO!=0).sum()) + "\n")
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n")

    sel_MLE = np.zeros(p)
    estimate, _, _, pval, intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(target=target, dispersion=dispersion)
    sel_MLE[nonzero] = estimate / np.sqrt(n)
    ind_estimator = np.zeros(p)
    ind_estimator[nonzero] = ind_unbiased_estimator / np.sqrt(n)

    return relative_risk(sel_MLE, beta, Sigma),\
           relative_risk(ind_estimator, beta, Sigma),\
           relative_risk(randomized_lasso.initial_soln / np.sqrt(n), beta, Sigma),\
           relative_risk(randomized_lasso._beta_full / np.sqrt(n), beta, Sigma), \
           relative_risk(rel_LASSO, beta, Sigma),\
           relative_risk(est_LASSO, beta, Sigma)

if __name__ == "__main__":

    ndraw = 50
    bias = 0.
    risk_selMLE = 0.
    risk_indest = 0.
    risk_LASSO_rand = 0.
    risk_relLASSO_rand = 0.

    risk_relLASSO_nonrand = 0.
    risk_LASSO_nonrand = 0.

    for i in range(ndraw):
        output = comparison_risk_inference(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=2, snr=0.2,
                                           randomizer_scale=np.sqrt(0.25), target="selected", full_dispersion=True)

        risk_selMLE += output[0]
        risk_indest += output[1]
        risk_LASSO_rand += output[2]
        risk_relLASSO_rand += output[3]
        risk_relLASSO_nonrand += output[4]
        risk_LASSO_nonrand += output[5]

        sys.stderr.write("overall selMLE risk " + str(risk_selMLE / float(i + 1)) + "\n")
        sys.stderr.write("overall indep est risk " + str(risk_indest / float(i + 1)) + "\n")
        sys.stderr.write("overall randomized LASSO est risk " + str(risk_LASSO_rand / float(i + 1)) + "\n")
        sys.stderr.write("overall relaxed rand LASSO est risk " + str(risk_relLASSO_rand / float(i + 1)) + "\n"+ "\n")

        sys.stderr.write("overall relLASSO risk " + str(risk_relLASSO_nonrand / float(i + 1)) + "\n")
        sys.stderr.write("overall LASSO risk " + str(risk_LASSO_nonrand / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("iteration completed" + str(i+1) + "\n")


