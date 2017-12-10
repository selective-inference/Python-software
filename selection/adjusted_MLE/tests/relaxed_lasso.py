from rpy2.robjects.packages import importr
from rpy2 import robjects

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np

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

def tuned_lasso(X, y, X_val,y_val):
    robjects.r('''
        source('~/best-subset/bestsubset/R/lasso.R')
        tuned_lasso_estimator = function(X,Y,X.val,Y.val){
        Y = as.matrix(Y)
        X = as.matrix(X)
        Y.val = as.vector(Y.val)
        X.val = as.matrix(X.val)

        rel.lasso = lasso(X,Y,intercept=FALSE, nrelax=10, nlam=50)
        beta.hat = as.matrix(coef(rel.lasso))

        muhat.val = as.matrix(predict(rel.lasso, X.val))
        err.val = colMeans((muhat.val - Y.val)^2)
        return(beta.hat[,which.min(err.val)])
        }''')

    r_lasso = robjects.globalenv['tuned_lasso_estimator']

    n, p = X.shape
    nval, _ = X_val.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    r_X_val = robjects.r.matrix(X_val, nrow=nval, ncol=p)
    r_y_val = robjects.r.matrix(y_val, nrow=nval, ncol=1)
    estimator = np.array(r_lasso(r_X, r_y, r_X_val, r_y_val))
    return estimator

X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=2, snr=0.2)
rel_LASSO = tuned_lasso(X, y, X_val,y_val)
print("relaxed LASSO", rel_LASSO)
