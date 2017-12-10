from rpy2.robjects.packages import importr
from rpy2 import robjects

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np

def sim_xy(n, p, nval, rho=0, s=5):
    robjects.r('''
    source('~/best-subset/bestsubset/R/sim.R')
    ''')

    r_simulate = robjects.globalenv['sim.xy']
    print(r_simulate(n, p, nval, rho=rho, s=s))

#sim_xy(n=50, p=10, nval=50)

def tuned_lasso(X, Y, X_val,Y_val):
    robjects.r('''
        source('~/best-subset/bestsubset/R/lasso.R')
        tuned_lasso_estimator = function(X,Y,X.val,Y.val){
        Y = as.matrix(Y)
        X = as.matrix(X)
        Y.val = as.vector(Y.val)
        X.val = as.matrix(X.val)

        rel.lasso = lasso(X,Y,intercept=TRUE, nrelax=5, nlam=50)
        beta.hat = as.matrix(coef(rel.lasso))

        muhat.val = as.matrix(predict(rel.lasso, X.val))
        err.val = colMeans((muhat.val - Y.val)^2)
        return(beta.hat[,which.min(err.val)])
        }''')

    r_lasso = robjects.globalenv['tuned_lasso_estimator']

    n, p = X.shape
    nval, _ = X_val.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(Y, nrow=n, ncol=1)

    r_X_val = robjects.r.matrix(X_val, nrow=nval, ncol=p)
    r_y_val = robjects.r.matrix(Y_val, nrow=nval, ncol=1)
    estimator = r_lasso(r_X, r_y, r_X_val, r_y_val)
    return (estimator)

print(tuned_lasso(np.random.standard_normal((50,10)), np.random.standard_normal(50),
                  np.random.standard_normal((50,10)), np.random.standard_normal(50)))