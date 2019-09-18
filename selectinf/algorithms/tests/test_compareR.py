from __future__ import print_function

import numpy as np, pandas as pd
import regreg.api as rr
import nose.tools as nt

try:
    import rpy2.robjects as rpy
    rpy2_available = True
    import rpy2.robjects.numpy2ri as numpy2ri
except ImportError:
    rpy2_available = False

try:
    import statsmodels.api as sm
    statsmodels_available = True
except ImportError:
    statsmodels_available = False

from ..lasso import lasso, ROSI
from ..forward_step import forward_step
from ...randomized.lasso import lasso as rlasso, selected_targets, full_targets, debiased_targets
from ...tests.instance import gaussian_instance, logistic_instance

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_fixed_lambda():
    """
    Check that Gaussian LASSO results agree with R
    """
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
        beta_hat = coef(gfit, s=lam/n, exact=TRUE, x=x, y=y)
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
    """
    Check that forward step results agree with R
    """
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
    """
    Check that forward step results agree with R
    """
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

@np.testing.dec.skipif(not rpy2_available or not statsmodels_available, msg="rpy2 not available, skipping test")
def test_coxph():
    """
    Check that Cox results agree with R
    """
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
    beta_hat = as.numeric(coef(gfit, s=lambda/n, exact=TRUE, x=x, y=Surv(tim, status)))
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
    """
    Check that logistic results agree with R
    """
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
    beta_hat = as.numeric(coef(gfit, s=lambda/n, exact=TRUE, x=x, y=y))

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


@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_solve_QP_lasso():
    """
    Check the R coordinate descent LASSO solver
    """

    n, p = 100, 200
    lam = 0.1

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    loss = rr.squared_error(X, Y, coef=1./n)
    pen = rr.l1norm(p, lagrange=lam)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve(min_its=500, tol=1.e-12)

    numpy2ri.activate()

    rpy.r.assign('X', X)
    rpy.r.assign('Y', Y)
    rpy.r.assign('lam', lam)

    R_code = """

    library(selectiveInference)
    p = ncol(X)
    n = nrow(X)
    soln_R = rep(0, p)
    grad = -t(X) %*% Y / n
    ever_active = as.integer(c(1, rep(0, p-1)))
    nactive = as.integer(1)
    kkt_tol = 1.e-12
    objective_tol = 1.e-16
    parameter_tol = 1.e-10
    maxiter = 500
    soln_R = selectiveInference:::solve_QP(t(X) %*% X / n, 
                                           lam, 
                                           maxiter, 
                                           soln_R, 
                                           1. * grad,
                                           grad, 
                                           ever_active, 
                                           nactive, 
                                           kkt_tol, 
                                           objective_tol, 
                                           parameter_tol,
                                           p,
                                           TRUE,
                                           TRUE,
                                           TRUE)$soln

    # test wide solver
    Xtheta = rep(0, n)
    nactive = as.integer(1)
    ever_active = as.integer(c(1, rep(0, p-1)))
    soln_R_wide = rep(0, p)
    grad = - t(X) %*% Y / n
    soln_R_wide = selectiveInference:::solve_QP_wide(X, 
                                                     rep(lam, p), 
                                                     0,
                                                     maxiter, 
                                                     soln_R_wide, 
                                                     1. * grad,
                                                     grad, 
                                                     Xtheta,
                                                     ever_active, 
                                                     nactive, 
                                                     kkt_tol, 
                                                     objective_tol, 
                                                     parameter_tol,
                                                     p,
                                                     TRUE,
                                                     TRUE,
                                                     TRUE)$soln

    """

    rpy.r(R_code)

    soln_R = np.asarray(rpy.r('soln_R'))
    soln_R_wide = np.asarray(rpy.r('soln_R_wide'))
    numpy2ri.deactivate()

    tol = 1.e-5
    print(soln - soln_R)
    print(soln_R - soln_R_wide)

    yield np.testing.assert_allclose, soln, soln_R, tol, tol, False, 'checking coordinate QP solver for LASSO problem'
    yield np.testing.assert_allclose, soln, soln_R_wide, tol, tol, False, 'checking wide coordinate QP solver for LASSO problem'

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_solve_QP():
    """
    Check the R coordinate descent LASSO solver
    """

    n, p = 100, 50
    lam = 0.08

    X = np.random.standard_normal((n, p))

    loss = rr.squared_error(X, np.zeros(n), coef=1./n)
    pen = rr.l1norm(p, lagrange=lam)
    E = np.zeros(p)
    E[2] = 1
    Q = rr.identity_quadratic(0, 0, E, 0)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve(Q, min_its=500, tol=1.e-12)

    numpy2ri.activate()

    rpy.r.assign('X', X)
    rpy.r.assign('E', E)
    rpy.r.assign('lam', lam)

    R_code = """

    library(selectiveInference)
    p = ncol(X)
    n = nrow(X)
    soln_R = rep(0, p)
    grad = 1. * E
    ever_active = as.integer(c(1, rep(0, p-1)))
    nactive = as.integer(1)
    kkt_tol = 1.e-12
    objective_tol = 1.e-16
    parameter_tol = 1.e-10
    maxiter = 500
    soln_R = selectiveInference:::solve_QP(t(X) %*% X / n, 
                                           lam, 
                                           maxiter, 
                                           soln_R, 
                                           E,
                                           grad, 
                                           ever_active, 
                                           nactive, 
                                           kkt_tol, 
                                           objective_tol, 
                                           parameter_tol,
                                           p,
                                           TRUE,
                                           TRUE,
                                           TRUE)$soln

    # test wide solver
    Xtheta = rep(0, n)
    nactive = as.integer(1)
    ever_active = as.integer(c(1, rep(0, p-1)))
    soln_R_wide = rep(0, p)
    grad = 1. * E
    soln_R_wide = selectiveInference:::solve_QP_wide(X, 
                                                     rep(lam, p), 
                                                     0,
                                                     maxiter, 
                                                     soln_R_wide, 
                                                     E,
                                                     grad, 
                                                     Xtheta,
                                                     ever_active, 
                                                     nactive, 
                                                     kkt_tol, 
                                                     objective_tol, 
                                                     parameter_tol,
                                                     p,
                                                     TRUE,
                                                     TRUE,
                                                     TRUE)$soln

    """

    rpy.r(R_code)

    soln_R = np.asarray(rpy.r('soln_R'))
    soln_R_wide = np.asarray(rpy.r('soln_R_wide'))
    numpy2ri.deactivate()

    tol = 1.e-5
    print(soln - soln_R)
    print(soln_R - soln_R_wide)

    G = X.T.dot(X).dot(soln) / n + E
    
    yield np.testing.assert_allclose, soln, soln_R, tol, tol, False, 'checking coordinate QP solver'
    yield np.testing.assert_allclose, soln, soln_R_wide, tol, tol, False, 'checking wide coordinate QP solver'
    yield np.testing.assert_allclose, G[soln != 0], -np.sign(soln[soln != 0]) * lam, tol, tol, False, 'checking active coordinate KKT for QP solver'
    yield nt.assert_true, np.fabs(G).max() < lam * (1. + 1.e-6), 'testing linfinity norm'

    
@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available")
def test_liu_gaussian():
    n, p, s = 200, 100, 20

    while True:

        X, y, _, _, sigma, _ = gaussian_instance(n=n, p=p, s=s, equicorrelated=False, signal=10, sigma=1.)

        lam = 4. * np.sqrt(n)
        X *= np.sqrt(n)
        L = ROSI.gaussian(X, y, lam, approximate_inverse=None)

        L.fit()
        if len(L.active) > 4:
            S = L.summary(compute_intervals=False, dispersion=sigma**2)
            numpy2ri.activate()

            rpy.r.assign('sigma_est', sigma)
            rpy.r.assign("X", X)
            rpy.r.assign("y", y)
            rpy.r.assign("lam", lam)
            rpy.r("""
            y = as.numeric(y)
            n = nrow(X)
            p = ncol(X)
            #sigma_est = sigma(lm(y ~ X - 1))
            penalty_factor = rep(1, p);
            soln = selectiveInference:::solve_problem_glmnet(X, 
                                                             y, 
                                                             lam/n, 
                                                             penalty_factor=penalty_factor,
                                                             family="gaussian")
            PVS = ROSI(X, 
                       y, 
                       soln, 
                       lambda=lam, 
                       penalty_factor=penalty_factor, 
                       dispersion=sigma_est^2, 
                       family="gaussian",
                       debiasing_method="JM",
                       solver="QP", 
                       construct_ci=FALSE,
                       use_debiased=FALSE)
            active_vars=PVS$active_vars - 1 # for 0-based
            pvalues = PVS$pvalues
            """)
            pvalues = np.asarray(rpy.r('pvalues'))
            pvalues = pvalues[~np.isnan(pvalues)]
            active_set = rpy.r('active_vars')

            print(pvalues)
            print(S['pval'])
            nt.assert_true(np.corrcoef(pvalues, S['pval'])[0,1] > 0.999)

            numpy2ri.deactivate()
            break

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available")
def test_liu_logistic():
    n, p, s = 200, 100, 20
    
    while True:

        X, y = logistic_instance(n=n, p=p, s=s, equicorrelated=False, signal=10)[:2]

        lam = 1. * np.sqrt(n)
        X *= np.sqrt(n)
        L = ROSI.logistic(X, y, lam, approximate_inverse=None)

        L.fit()
        if len(L.active) > 4:
            S = L.summary(compute_intervals=False, dispersion=1)
            numpy2ri.activate()

            rpy.r.assign("X", X)
            rpy.r.assign("y", y)
            rpy.r.assign("lam", lam)
            rpy.r("""
            y = as.numeric(y)
            n = nrow(X)
            p = ncol(X)
            sigma_est = 1
            print(sigma_est)
            penalty_factor = rep(1, p);
            soln = selectiveInference:::solve_problem_glmnet(X, 
                                                             y, 
                                                             lam/n, 
                                                             penalty_factor=penalty_factor,
                                                             family="binomial")
            PVS = ROSI(X, 
                       y, 
                       soln, 
                       lambda=lam, 
                       penalty_factor=penalty_factor, 
                       dispersion=1,
                       family="binomial",
                       debiasing_method="JM",
                       solver="Q", 
                       construct_ci=FALSE,
                       use_debiased=FALSE)
            active_vars=PVS$active_vars - 1 # for 0-based
            pvalues = PVS$pvalues
            """)
            pvalues = np.asarray(rpy.r('pvalues'))
            pvalues = pvalues[~np.isnan(pvalues)]
            active_set = rpy.r('active_vars')
            print(pvalues)
            print(S['pval'])
            nt.assert_true(np.corrcoef(pvalues, S['pval'])[0,1] > 0.999)

            numpy2ri.deactivate()
            break 

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available")
def test_ROSI_gaussian_JM():
    n, p, s = 100, 30, 15

    while True:
        X, y, _, _, sigma, _ = gaussian_instance(n=n, p=p, s=s, equicorrelated=False, signal=4)

        lam = 7. * np.sqrt(n)
        X *= np.sqrt(n)
        L = ROSI.gaussian(X, y, lam, approximate_inverse='JM')
        L.sparse_inverse = True
        L.fit()

        print('here', len(L.active))
        if len(L.active) > 4:
            S = L.summary(compute_intervals=False, 
                          dispersion=sigma**2)
            numpy2ri.activate()

            rpy.r.assign("X", X)
            rpy.r.assign("y", y)
            rpy.r.assign("sigma_est", sigma)
            rpy.r.assign("lam", lam)
            rpy.r("""

            y = as.numeric(y)
            n = nrow(X)
            p = ncol(X)

            penalty_factor = rep(1, p);
            soln = selectiveInference:::solve_problem_glmnet(X, 
                                                             y, 
                                                             lam/n, 
                                                             penalty_factor=penalty_factor,
                                                             family="gaussian")
            PVS = ROSI(X, 
                       y, 
                       soln, 
                       lambda=lam, 
                       penalty_factor=penalty_factor, 
                       dispersion=sigma_est^2, 
                       family="gaussian",
                       solver="QP", 
                       construct_ci=FALSE,  
                       use_debiased=TRUE)
            active_vars=PVS$active_vars - 1 # for 0-based
            pvalues = PVS$pvalues
            """)
            pvalues = np.asarray(rpy.r('pvalues'))
            pvalues = pvalues[~np.isnan(pvalues)]
            active_set = rpy.r('active_vars')

            print(pvalues)
            print(np.asarray(S['pval']))

            nt.assert_true(np.corrcoef(pvalues, S['pval'])[0,1] > 0.999)
            numpy2ri.deactivate()
            break

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available")
def test_ROSI_logistic_JM():
    n, p, s = 100, 30, 15

    while True:
        X, y = logistic_instance(n=n, p=p, s=s, equicorrelated=False, signal=10)[:2]

        lam = 1. * np.sqrt(n)
        X *= np.sqrt(n)
        L = ROSI.logistic(X, y, lam, approximate_inverse='JM')
        L.fit()

        if len(L.active) > 4:
            S = L.summary(compute_intervals=False, dispersion=1.)
            numpy2ri.activate()

            rpy.r.assign("X", X)
            rpy.r.assign("y", y)
            rpy.r.assign("lam", lam)
            rpy.r("""

            y = as.numeric(y)
            n = nrow(X)
            p = ncol(X)

            penalty_factor = rep(1, p);
            soln = selectiveInference:::solve_problem_glmnet(X, 
                                                             y, 
                                                             lam/n, 
                                                             penalty_factor=penalty_factor,
                                                             family="binomial")
            PVS = ROSI(X, 
                       y, 
                       soln, 
                       lambda=lam, 
                       penalty_factor=penalty_factor, 
                       dispersion=1., 
                       family="binomial", 
                       solver="QP", 
                       construct_ci=FALSE,
                       use_debiased=TRUE)
            active_vars=PVS$active_vars - 1 # for 0-based
            pvalues = PVS$pvalues
            """)
            pvalues = np.asarray(rpy.r('pvalues'))
            pvalues = pvalues[~np.isnan(pvalues)]
            active_set = rpy.r('active_vars')

            print(pvalues)
            print(np.asarray(S['pval']))

            nt.assert_true(np.corrcoef(pvalues, S['pval'])[0,1] > 0.999)
            numpy2ri.deactivate()
            break

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available")
def test_ROSI_gaussian_BN():
    n, p, s = 200, 500, 20

    while True:
        X, y, _, _, sigma, _ = gaussian_instance(n=n, p=p, s=s, equicorrelated=False, signal=3.5, sigma=1)

        lam = np.sqrt(n * 2 * np.log(p))
        X *= np.sqrt(n)
        L = ROSI.gaussian(X, y, lam, approximate_inverse='BN')
        L.sparse_inverse = True
        L.fit()

        print('here', len(L.active))
        if len(L.active) > 4:

            df = pd.DataFrame(X, 
                              columns=['X%d' % i for i in range(1, X.shape[1]+1)])
            df['y'] = y
            df.to_csv('data.csv', index=False)

            S = L.summary(compute_intervals=False, 
                          dispersion=sigma**2)
            numpy2ri.activate()

            rpy.r.assign("X", X)
            rpy.r.assign("y", y)
            rpy.r.assign("sigma_est", sigma)
            rpy.r.assign("lam", lam)
            rpy.r("""

            y = as.numeric(y)
            n = nrow(X)
            p = ncol(X)

            penalty_factor = rep(1, p);
            soln = selectiveInference:::solve_problem_glmnet(X, 
                                                             y, 
                                                             lam/n, 
                                                             penalty_factor=penalty_factor,
                                                             family="gaussian")
            PVS = ROSI(X, 
                       y, 
                       soln, 
                       lambda=lam, 
                       penalty_factor=penalty_factor, 
                       dispersion=sigma_est^2, 
                       family="gaussian",
                       debiasing_method="BN",
                       solver="QP", 
                       construct_ci=FALSE,  
                       use_debiased=TRUE)
            active_vars=PVS$active_vars - 1 # for 0-based
            pvalues = PVS$pvalues
            """)
            pvalues = np.asarray(rpy.r('pvalues'))
            pvalues = pvalues[~np.isnan(pvalues)]
            active_set = rpy.r('active_vars')

            print(pvalues)
            print(np.asarray(S['pval']))

            nt.assert_true(np.corrcoef(pvalues, S['pval'])[0,1] > 0.999)
            numpy2ri.deactivate()
            break

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available")
def test_ROSI_logistic_BN():
    n, p, s = 100, 120, 15

    while True:
        X, y = logistic_instance(n=n, p=p, s=s, equicorrelated=False, signal=10)[:2]

        lam = 1. * np.sqrt(n)
        X *= np.sqrt(n)
        L = ROSI.logistic(X, y, lam, approximate_inverse='BN')
        L.fit()

        if len(L.active) > 4:
            S = L.summary(compute_intervals=False, dispersion=1.)
            numpy2ri.activate()

            rpy.r.assign("X", X)
            rpy.r.assign("y", y)
            rpy.r.assign("lam", lam)
            rpy.r("""

            y = as.numeric(y)
            n = nrow(X)
            p = ncol(X)

            penalty_factor = rep(1, p);
            soln = selectiveInference:::solve_problem_glmnet(X, 
                                                             y, 
                                                             lam/n, 
                                                             penalty_factor=penalty_factor,
                                                             family="binomial")
            PVS = ROSI(X, 
                       y, 
                       soln, 
                       lambda=lam, 
                       penalty_factor=penalty_factor, 
                       dispersion=1., 
                       family="binomial", 
                       debiasing_method="BN",
                       solver="QP", 
                       construct_ci=FALSE,
                       use_debiased=TRUE)
            active_vars=PVS$active_vars - 1 # for 0-based
            pvalues = PVS$pvalues
            """)
            pvalues = np.asarray(rpy.r('pvalues'))
            pvalues = pvalues[~np.isnan(pvalues)]
            active_set = rpy.r('active_vars')

            print(pvalues)
            print(np.asarray(S['pval']))

            nt.assert_true(np.corrcoef(pvalues, S['pval'])[0,1] > 0.999)
            numpy2ri.deactivate()
            break

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available")
def test_rlasso_gaussian():
    """
    Check that the randomized results agree with 
    R given same inputs, randomization & samples
    """
    n, p, s, sigma, signal_fac, rho = 100, 30, 15, 3, 1.5, 0.4
    randomizer_scale, ndraw, burnin = 1, 5000, 1000
    target = 'selected'
    R_target = 'selected'

    while True:
        signal = np.sqrt(signal_fac * np.log(p))
        X, y, beta, active, sigma, _ = gaussian_instance(n=n,
                                                         p=p, 
                                                         signal=signal, 
                                                         s=s, 
                                                         equicorrelated=False, 
                                                         rho=rho, 
                                                         sigma=sigma, 
                                                         random_signs=True)

        sigma_ = np.std(y)
        if target is not 'debiased':
            lam = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma_
        else:
            lam = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_
        L = rlasso.gaussian(X,y,lam)
        ridge_term = L.ridge_term
        _,prec = L.randomizer.cov_prec
        noise_scale = np.sqrt(1./prec)

        numpy2ri.activate()

        rpy.r.assign("X", X)
        rpy.r.assign("y", y)
        rpy.r.assign("lam", lam)
        rpy.r.assign("ridge_term", ridge_term)
        rpy.r.assign("type", R_target)
        rpy.r.assign("noise_scale",noise_scale)
        rpy.r("""

        library(selectiveInference)

        y = as.numeric(y)
        n = nrow(X)
        p = ncol(X)

        penalty_factor = rep(1, p);
        soln = randomizedLasso(X, 
                               y, 
                               lam, 
                               family="gaussian",
                               noise_scale=noise_scale,
                               ridge_term=ridge_term)
        perturb = soln$perturb
        obs_soln = soln$soln
        cond_mean = soln$law$cond_mean
        cond_cov = soln$law$cond_cov

        if (type != 'selected') {
            targets = selectiveInference:::compute_target(soln, type)
        } else {
            targets = NULL
        }

        PVS = randomizedLassoInf(soln,
                                 targets=targets,
                                 sampler="norejection")
        opt_samples = PVS$opt_samples
        target_samples = PVS$target_samples
        pvalues = PVS$pvalues
        """)
        R_perturb = rpy.r('perturb')

        R_soln = rpy.r('obs_soln')
        R_cond_mean = rpy.r('cond_mean')
        R_cond_mean = np.squeeze(R_cond_mean)
        R_cond_cov = rpy.r('cond_cov')

        R_pvalues = np.asarray(rpy.r('pvalues'))
        R_pvalues = R_pvalues[~np.isnan(R_pvalues)]

        R_opt_samples = rpy.r('opt_samples')
        R_target_samples = rpy.r('target_samples')

        numpy2ri.deactivate()

        signs = L.fit(perturb=np.asarray(R_perturb))
        active_set = np.where(signs != 0)[0]
        nonzero = signs != 0

        initial_soln = L.initial_soln
        cond_mean = L.cond_mean
        cond_cov = L.cond_cov

        if nonzero.sum() > 4:
            if target == 'full':
                (observed_target, 
                 cov_target, 
                 cov_target_score, 
                 alternatives) = full_targets(L.loglike, 
                                              L._W, 
                                              nonzero)
            elif target == 'selected':
                (observed_target, 
                 cov_target, 
                 cov_target_score, 
                 alternatives) = selected_targets(L.loglike, 
                                                  L._W, 
                                                  nonzero)
            elif target == 'debiased':
                (observed_target, 
                 cov_target, 
                 cov_target_score, 
                 alternatives) = debiased_targets(L.loglike, 
                                                  L._W, 
                                                  nonzero,
                                                  penalty=L.penalty)

            _, pval, intervals = L.summary(observed_target, 
                                           cov_target, 
                                           cov_target_score, 
                                           alternatives,
                                           opt_sample=(np.asarray(R_opt_samples),),
                                           target_sample=np.asarray(R_target_samples),
                                           ndraw=8000,#ndraw,
                                           burnin=burnin, 
                                           compute_intervals=True)

            tol = 1.e-5
            yield np.testing.assert_allclose, initial_soln, R_soln, tol, tol, False, 'checking initial rlasso solution'
            yield np.testing.assert_allclose, cond_mean, R_cond_mean, tol, tol, False, 'checking conditional mean'
            yield np.testing.assert_allclose, cond_cov, R_cond_cov, tol, tol, False, 'checking conditional covariance'
            yield np.testing.assert_allclose, pval, R_pvalues, tol, tol, False, 'checking pvalues'

            break


        
