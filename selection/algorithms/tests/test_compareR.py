from __future__ import print_function

import numpy as np
import nose.tools as nt

import rpy2.robjects as rpy
from selection.algorithms.lasso import lasso
from selection.algorithms.forward_step import forward_step

def test_fixed_lambda():
    R_code = """
    library(selectiveInference)
    set.seed(43)
    n = 50
    p = 10
    sigma = 1
    x = matrix(rnorm(n*p),n,p)
    x=scale(x,TRUE,TRUE)
    beta = c(3,2,rep(0,p-2))
    y = x%*%beta + sigma*rnorm(n)
    # first run glmnet
    gfit = glmnet(x,y,standardize=FALSE)
    # extract coef for a given lambda; note the 1/n factor!
    # (and we don't save the intercept term)
    lambda = .8
    beta_hat = coef(gfit, s=lambda/n, exact=TRUE)[-1]

    # compute fixed lambda p-values and selection intervals
    pval = fixedLassoInf(x,y,beta_hat,lambda,sigma=sigma)$pv
    """

    rpy.r(R_code)
    R_pvals = np.asarray(rpy.r('pval'))
    y = np.asarray(rpy.r('y'))
    beta_hat = np.asarray(rpy.r('beta_hat'))
    x = np.asarray(rpy.r('x'))
    y = y.reshape(-1)
    y -= y.mean()
    x -= x.mean(0)[None,:]
    L = lasso.gaussian(x, y, 0.8, sigma=1)

    np.testing.assert_allclose(L.fit(), beta_hat, atol=1.e-3, rtol=1.)
    np.testing.assert_allclose([p for _, p in L.onesided_pvalues], R_pvals, atol=1.e-4, rtol=1.)


def test_forward_step():
    R_code = """
    set.seed(33)
    n = 50
    p = 10
    sigma = 1
    x = matrix(rnorm(n*p),n,p)
    beta = c(3,2,rep(0,p-2))
    y = x%*%beta + sigma*rnorm(n)
    # run forward stepwise
    fsfit = fs(x,y)
    beta_hat = fsfit$beta
    # compute sequential p-values and confidence intervals
    # (sigma estimated from full model)
    out.seq = fsInf(fsfit,sigma=1)
    vars = out.seq$vars
    pval = out.seq$pv
    """

    rpy.r(R_code)
    R_pvals = np.asarray(rpy.r('pval'))
    selected_vars = np.asarray(rpy.r('vars'))
    y = np.asarray(rpy.r('y'))
    beta_hat = np.asarray(rpy.r('beta_hat'))
    x = np.asarray(rpy.r('x'))
    y = y.reshape(-1)
    y -= y.mean()
    x -= x.mean(0)[None,:]
    sigma = 1
    FS = forward_step(x, y, covariance=sigma**2 * np.identity(y.shape[0]))
    steps = []
    for i in range(x.shape[1]):
        FS.next()
        steps.extend(FS.model_pivots(i+1, which_var=FS.variables[-1:]))

    np.testing.assert_array_equal(selected_vars, [i + 1 for i, p in steps])
    np.testing.assert_allclose([p for i, p in steps], R_pvals, atol=1.e-4, rtol=1.)
