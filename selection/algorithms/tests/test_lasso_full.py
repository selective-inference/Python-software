from __future__ import print_function

import numpy as np, regreg.api as rr

from ...tests.instance import gaussian_instance

from ..lasso import (lasso_full,
                     lasso_full_modelQ,
                     _truncation_interval,
                     _solve_restricted_problem)

# earlier implmentation

def solve_problem(Qbeta_bar, Q, lagrange, initial=None):
    p = Qbeta_bar.shape[0]
    loss = rr.quadratic_loss((p,), Q=Q, quadratic=rr.identity_quadratic(0, 
                                                                        -Qbeta_bar, 
                                                                        0))
    lagrange = np.asarray(lagrange)
    if lagrange.shape in [(), (1,)]:
        lagrange = np.ones(p) * lagrange
    pen = rr.weighted_l1norm(lagrange, lagrange=1.)
    problem = rr.simple_problem(loss, pen)
    if initial is not None:
        problem.coefs[:] = initial
    soln = problem.solve(tol=1.e12, min_its=100)

    return soln

def truncation_interval(Qbeta_bar, Q, Qi_jj, j, beta_barj, lagrange):
    if lagrange[j] != 0:
        lagrange_cp = lagrange.copy()
    lagrange_cp[j] = np.inf
    restricted_soln = solve_problem(Qbeta_bar, Q, lagrange_cp)

    p = Qbeta_bar.shape[0]
    I = np.identity(p)
    nuisance = Qbeta_bar - I[:,j] / Qi_jj * beta_barj
    center = nuisance[j] - Q[j].dot(restricted_soln)
    upper = (lagrange[j] - center) * Qi_jj
    lower = (-lagrange[j] - center) * Qi_jj

    return lower, upper

def test_smaller():

    n, p, s = 200, 100, 4
    X, y, beta = gaussian_instance(n=n,
                                   p=p,
                                   s=s)[:3]

    lagrange = 10. * np.ones(p)

    LF = lasso_full.gaussian(X, y, lagrange)
    LF.fit()

    Q = X.T.dot(X)
    Qbeta_bar = X.T.dot(y)
    beta_hat = solve_problem(Qbeta_bar, Q, lagrange)
    beta_hat2 = _solve_restricted_problem(Qbeta_bar, X, np.ones(X.shape[0]), 
                                          lagrange, min_its=100)

    Qi = np.linalg.inv(Q)
    beta_bar = np.linalg.pinv(X).dot(y)
    sigma = np.linalg.norm(y - X.dot(beta_bar)) / np.sqrt(n - p)

    E = LF.active
    QiE = Qi[E][:,E]
    beta_barE = beta_bar[E]

    S = LF.summary()

    for i, j in enumerate(LF.active):
        l, u = (np.array(S['lower_truncation'])[i], 
                np.array(S['upper_truncation'])[i]) 
        lower, upper =  truncation_interval(Qbeta_bar, Q, QiE[i,i], j, beta_barE[i], lagrange)
        np.testing.assert_allclose(l, lower)
        np.testing.assert_allclose(u, upper)

def test_modelQ():

    n, p, s = 200, 50, 4
    X, y, beta = gaussian_instance(n=n,
                                   p=p,
                                   s=s,
                                   sigma=1)[:3]

    lagrange = 1. * np.ones(p)

    LF = lasso_full.gaussian(X, y, lagrange)
    LF.fit()
    S = LF.summary(dispersion=1)

    LX = lasso_full_modelQ(X.T.dot(X), X, y, lagrange)
    LX.fit()
    SX = LX.summary(dispersion=1)

    np.testing.assert_allclose(S['pval'], SX['pval'])




