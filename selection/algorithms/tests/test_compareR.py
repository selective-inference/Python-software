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


@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_fixed_lambda():
    tol = 1.e-5
    for s in [1,1.1]:
        lam = 7.8
        R_code = """

        library(selectiveInference)
        set.seed(43)
        n = 50
        p = 10
        sigma = %f

        x = matrix(rnorm(n*p),n,p)
        x=scale(x,TRUE,TRUE)

        beta = c(3,-2,rep(0,p-2))
        y = x%%*%%beta + sigma*rnorm(n)

        # first run glmnet
        gfit = glmnet(x,y,standardize=FALSE)

        # extract coef for a given lambda; note the 1/n factor!
        # (and we don't save the intercept term)
        lam = %f
        beta_hat = coef(gfit, s=lam/n, exact=TRUE)
        beta_hat = beta_hat[-1]

        # compute fixed lambda p-values and selection intervals
        out = fixedLassoInf(x,y,beta_hat,lam,sigma=sigma)

        vlo = out$vlo
        vup = out$vup

        sdvar = out$sd
        pval=out$pv
        coef0=out$coef0
        vars=out$vars
        print(coef(lm(y ~ x[,out$vars])))
        out 
        """ % (s, lam)

        rpy.r(R_code)

        R_pvals = np.asarray(rpy.r('pval'))
        selected_vars = np.asarray(rpy.r('vars'))
        coef = np.asarray(rpy.r('coef0')).reshape(-1)
        sdvar = np.asarray(rpy.r('sdvar'))
        y = np.asarray(rpy.r('y'))
        beta_hat = np.asarray(rpy.r('as.numeric(beta_hat)'))
        x = np.asarray(rpy.r('x'))
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        y = y.reshape(-1)
        #y -= y.mean()
        L = lasso.gaussian(x, y, lam, sigma=s)
        L.fit(solve_args={'min_its':200})

        S = L.summary('onesided')
        yield np.testing.assert_allclose, L.fit()[1:], beta_hat, 1.e-2, 1.e-2, False, 'fixed lambda, sigma=%f coef' % s
        yield np.testing.assert_equal, L.active, selected_vars
        yield np.testing.assert_allclose, S['pval'], R_pvals, tol, tol, False, 'fixed lambda, sigma=%f pval' % s
        yield np.testing.assert_allclose, S['sd'], sdvar, tol, tol, False, 'fixed lambda, sigma=%f sd ' % s
        yield np.testing.assert_allclose, S['onestep'], coef, tol, tol, False, 'fixed lambda, sigma=%f estimator' % s

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_forward_step():
    tol = 1.e-5
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
    vlo = out.seq$vlo
    vup = out.seq$vup
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

    vlo = np.asarray(rpy.r('vlo'))
    vup = np.asarray(rpy.r('vup'))
    print(np.vstack([vlo, vup]).T)
    FS = forward_step(x, y, covariance=sigma**2 * np.identity(y.shape[0]))
    steps = []
    for i in range(x.shape[1]):
        FS.step()
        steps.extend(FS.model_pivots(i+1, 
                                     which_var=FS.variables[-1:],
                                     alternative='onesided'))

    print(selected_vars, [i+1 for i, p in steps])
    print(FS.variables, FS.signs)
    np.testing.assert_array_equal(selected_vars, [i + 1 for i, p in steps])
    np.testing.assert_allclose([p for i, p in steps], R_pvals, atol=tol, rtol=tol)

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_forward_step_all():
    tol = 1.e-5
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

    vlo = np.asarray(rpy.r('vlo'))
    vup = np.asarray(rpy.r('vup'))
    print(np.vstack([vlo, vup]).T)
    FS = forward_step(x, y, covariance=sigma**2 * np.identity(y.shape[0]))
    steps = []
    for i in range(5):
        FS.step()
    steps = FS.model_pivots(5, 
                            alternative='onesided')

    np.testing.assert_array_equal(selected_vars, [i + 1 for i, p in steps])
    np.testing.assert_allclose([p for i, p in steps], R_pvals, atol=tol, rtol=tol)

    print (R_pvals, [p for i, p in steps])

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_coxph():
    tol = 1.e-5
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


    gfit = glmnet(x,Surv(tim,status),standardize=FALSE,family="cox", thresh=1.e-14)
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

    G1 = L.loglike.gradient(beta_hat)
    G2 = L.loglike.gradient(beta2)

    print(G1, 'glmnet')
    print(G2, 'regreg')

    yield np.testing.assert_equal, np.array(L.active) + 1, selected_vars
    yield np.testing.assert_allclose, beta2, beta_hat, tol, tol, False, 'cox coeff'
    yield np.testing.assert_allclose, L.summary('onesided')['pval'], R_pvals, tol, tol, False, 'cox pvalues'

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_logistic():
    tol = 1.e-4
    R_code = """
    library(selectiveInference)
    set.seed(43)
    n = 50
    p = 10
    sigma = 10

    x = matrix(rnorm(n*p),n,p)
    x=scale(x,TRUE,TRUE)

    beta = c(3,2,rep(0,p-2))
    y = x %*% beta + sigma * rnorm(n)
    y=1*(y>mean(y))
    # first run glmnet
    gfit = glmnet(x,y,standardize=FALSE,family="binomial")

    # extract coef for a given lambda; note the 1/n factor!
    # (and here  we DO  include the intercept term)
    lambda = .8
    beta_hat = as.numeric(coef(gfit, s=lambda/n, exact=TRUE))

    # compute fixed lambda p-values and selection intervals
    out = fixedLassoInf(x,y,beta_hat,lambda,family="binomial")
    vlo = out$vlo
    vup = out$vup
    sdvar = out$sd
    coef=out$coef0
    info_mat=out$info.matrix
    beta_hat = beta_hat[c(1, out$vars+1)]
    out
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
    beta2 = L.fit()[L.active]

    yield np.testing.assert_equal, L.active[1:], selected_vars
    yield np.testing.assert_allclose, beta2, beta_hat, tol, tol, False, 'logistic coef'
    yield np.testing.assert_allclose, L.summary('onesided')['pval'][1:], R_pvals, tol, tol, False, 'logistic pvalues'


