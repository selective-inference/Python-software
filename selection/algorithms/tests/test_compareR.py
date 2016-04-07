from __future__ import print_function

import numpy as np
import nose.tools as nt

try:
    import rpy2.robjects as rpy
    rpy2_available = True
except ImportError:
    rpy2_available = False

from selection.algorithms.lasso import lasso
from selection.algorithms.forward_step import forward_step

tol = 1.e-2

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_fixed_lambda():
    for s in [1,1.1]:
        R_code = """
        library(selectiveInference)
        set.seed(43)
        n = 50
        p = 10
        sigma = %f
        x = matrix(rnorm(n*p),n,p)
        x = scale(x,TRUE,TRUE)
        beta = c(3,2,rep(0,p-2))
        y = x%%*%%beta + sigma*rnorm(n)
        # first run glmnet
        gfit = glmnet(x,y,standardize=FALSE)
        # extract coef for a given lambda; note the 1/n factor!
        # (and we don't save the intercept term)
        lambda = .8
        beta_hat = coef(gfit, s=lambda/n, exact=TRUE)[-1]

        # compute fixed lambda p-values and selection intervals
        out = fixedLassoInf(x,y,beta_hat,lambda,sigma=sigma)
        pval = out$pv
        vars = out$var
        """ % s

        rpy.r(R_code)
        R_pvals = np.asarray(rpy.r('pval'))
        sigma = float(np.asarray(rpy.r('sigma')))
        selected_vars = np.asarray(rpy.r('vars'))
        y = np.asarray(rpy.r('y'))
        beta_hat = np.asarray(rpy.r('as.numeric(beta_hat)'))
        x = np.asarray(rpy.r('x'))
        y = y.reshape(-1)
        y -= y.mean()
        x -= x.mean(0)[None,:]
        L = lasso.gaussian(x, y, 0.8, sigma=sigma)
        L.fit()

        yield np.testing.assert_allclose, L.fit(), beta_hat, tol, tol, False, 'fixed lambda, sigma=%f' % s
        yield np.testing.assert_equal, L.active + 1, selected_vars
        yield np.testing.assert_allclose, [p for _, p in L.onesided_pvalues], R_pvals, tol, tol, False, 'fixed lambda, sigma=%f' % s

        print([p for _, p in L.onesided_pvalues], R_pvals)

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_forward_step():
    R_code = """
    library(selectiveInference)
    set.seed(33)
    n = 50
    p = 10
    sigma = 1.1
    x = matrix(rnorm(n*p),n,p)
    beta = c(3,2,rep(0,p-2))
    y = x%*%beta + sigma*rnorm(n)
    # run forward stepwise
    fsfit = fs(x,y)
    beta_hat = fsfit$beta
    # compute sequential p-values and confidence intervals
    out.seq = fsInf(fsfit,sigma=sigma)
    vars = out.seq$vars
    pval = out.seq$pv
    """

    rpy.r(R_code)
    R_pvals = np.asarray(rpy.r('pval'))
    sigma = float(np.asarray(rpy.r('sigma')))
    selected_vars = np.asarray(rpy.r('vars'))
    y = np.asarray(rpy.r('y'))
    beta_hat = np.asarray(rpy.r('beta_hat'))
    x = np.asarray(rpy.r('x'))
    y = y.reshape(-1)
    y -= y.mean()
    x -= x.mean(0)[None,:]
    FS = forward_step(x, y, covariance=sigma**2 * np.identity(y.shape[0]))
    steps = []
    for i in range(x.shape[1]):
        FS.next()
        steps.extend(FS.model_pivots(i+1, 
                                     which_var=FS.variables[-1:],
                                     alternative='onesided'))

    np.testing.assert_array_equal(selected_vars, [i + 1 for i, p in steps])
    np.testing.assert_allclose([p for i, p in steps], R_pvals, atol=tol, rtol=tol)

    print (R_pvals, [p for i, p in steps])

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_forward_step_all():
    R_code = """
    library(selectiveInference)
    set.seed(33)
    n = 50
    p = 10
    sigma = 1.1
    x = matrix(rnorm(n*p),n,p)
    beta = c(3,2,rep(0,p-2))
    y = x%*%beta + sigma*rnorm(n)
    # run forward stepwise
    fsfit = fs(x,y)
    beta_hat = fsfit$beta
    # compute sequential p-values and confidence intervals
    out.seq = fsInf(fsfit,sigma=sigma, type='all', k=5)
    vars = out.seq$vars
    pval = out.seq$pv
    """

    rpy.r(R_code)
    R_pvals = np.asarray(rpy.r('pval'))
    sigma = float(np.asarray(rpy.r('sigma')))
    selected_vars = np.asarray(rpy.r('vars'))
    y = np.asarray(rpy.r('y'))
    beta_hat = np.asarray(rpy.r('beta_hat'))
    x = np.asarray(rpy.r('x'))
    y = y.reshape(-1)
    y -= y.mean()
    x -= x.mean(0)[None,:]
    FS = forward_step(x, y, covariance=sigma**2 * np.identity(y.shape[0]))
    steps = []
    for i in range(5):
        FS.next()
    steps = FS.model_pivots(5, 
                            alternative='onesided')

    np.testing.assert_array_equal(selected_vars, [i + 1 for i, p in steps])
    np.testing.assert_allclose([p for i, p in steps], R_pvals, atol=tol, rtol=tol)

    print (R_pvals, [p for i, p in steps])

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_coxph():
    R_code = """
    library(selectiveInference)
    set.seed(43)
    n = 50
    p = 10
    sigma = 1.1

    x = matrix(rnorm(n*p),n,p)
    x=scale(x,TRUE,TRUE)

    beta = c(3,2,rep(0,p-2))
    tim = as.vector(x%*%beta + sigma*rnorm(n))
    tim= tim-min(tim)+1
    status=sample(c(0,1),size=n,replace=T)
    # first run glmnet


    gfit = glmnet(x,Surv(tim,status),standardize=FALSE,family="cox")
    # extract coef for a given lambda; note the 1/n factor!

    lambda = 1.5
    beta_hat = as.numeric(coef(gfit, s=lambda/n, exact=TRUE))
    # compute fixed lambda p-values and selection intervals
    out = fixedLassoInf(x,tim,beta_hat,lambda,status=status,family="cox")
    pval = out$pv
    vars_cox = out$var


    """

    rpy.r(R_code)
    R_pvals = np.asarray(rpy.r('pval'))
    selected_vars = np.asarray(rpy.r('vars_cox'))
    tim = np.asarray(rpy.r('tim'))
    tim = tim.reshape(-1)

    status = np.asarray(rpy.r('status'))
    status = status.reshape(-1)

    beta_hat = np.asarray(rpy.r('beta_hat'))
    x = np.asarray(rpy.r('x'))

    L = lasso.coxph(x, tim, status, 1.5)
    beta2 = L.fit()

    yield np.testing.assert_equal, L.active + 1, selected_vars
    yield np.testing.assert_allclose, L.fit(), beta_hat, tol, tol, False, 'cox coeff'
    yield np.testing.assert_allclose, [p for _, p in L.active_pvalues], R_pvals, tol, tol, False, 'cox pvalues'

    print (R_pvals, [p for _, p in L.onesided_pvalues[1:]])

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_logistic():
    R_code = """
    library(selectiveInference)
    set.seed(43)
    n = 50
    p = 10
    sigma = 1.1

    x = matrix(rnorm(n*p),n,p)
    x=scale(x,TRUE,TRUE)
    beta = c(3,2,rep(0,p-2))
    y = sigma*x%*%beta + sigma*rnorm(n)
    y=1*(y>mean(y))
    # first run glmnet
    gfit = glmnet(x,y,standardize=FALSE,family="binomial")

    # extract coef for a given lambda; note the 1/n factor!
    # (and here  we DO  include the intercept term)
    lambda = .8
    beta_hat = coef(gfit, s=lambda/n, exact=TRUE)

    # compute fixed lambda p-values and selection intervals
    out = fixedLassoInf(x,y,beta_hat,lambda,family="binomial")
    pval = out$pv
    vars_logit = out$var

    """

    rpy.r(R_code)
    R_pvals = np.asarray(rpy.r('pval'))
    selected_vars = np.asarray(rpy.r('vars_logit'))

    y = np.asarray(rpy.r('y'))
    y = y.reshape(-1)

    beta_hat = np.asarray(rpy.r('as.numeric(beta_hat)'))
    x = np.asarray(rpy.r('x'))
    x = np.hstack([np.ones((x.shape[0],1)), x])
    L = lasso.logistic(x, y, [0] + [0.8] * (x.shape[1]-1))
    beta2 = L.fit()

    yield np.testing.assert_equal, L.active[1:], selected_vars
    yield np.testing.assert_allclose, beta2, beta_hat, tol, tol, False, 'logistic coef'
    yield np.testing.assert_allclose, [p for _, p in L.active_pvalues[1:]], R_pvals, tol, tol, False, 'logistic pvalues'

    print (R_pvals, [p for _, p in L.onesided_pvalues[1:]])
