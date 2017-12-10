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

def tuned_lasso(X, Y):
    robjects.r('''
        source('~/best-subset/bestsubset/R/lasso.R')
        tuned_lasso_estimator = function(X,Y){
        Y = as.matrix(Y)
        X = as.matrix(X)
        rel.lasso = lasso(X,Y,intercept=FALSE, nrelax=5, nlam=50)
        beta.hat = as.matrix(coef(rel.lasso))
        return(beta.hat)
        }''')

    r_lasso = robjects.globalenv['tuned_lasso_estimator']

    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(Y, nrow=n, ncol=1)

    estimator = r_lasso(r_X, r_y)
    return (estimator)

print(tuned_lasso(np.random.standard_normal((50,10)), np.random.standard_normal(50)))