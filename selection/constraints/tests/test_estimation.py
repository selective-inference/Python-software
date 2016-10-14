from __future__ import print_function
import numpy as np

import regreg.api as rr
from selection.constraints.estimation import (softmax,
                                              softmax_conjugate,
                                              gaussian_cumulant,
                                              gaussian_cumulant_conjugate,
                                              gaussian_cumulant_known,
                                              gaussian_cumulant_conjugate_known)

from selection.tests.flags import SET_SEED
from selection.tests.decorators import set_seed_iftrue
from selection.constraints.affine import constraints

@set_seed_iftrue(SET_SEED)
def test_softmax():

    A = np.array([[-1.,0.]])
    b = np.array([-1.])

    con = constraints(A, b)
    observed = np.array([1.4,2.3])

    simple_estimator = con.estimate_mean(observed)
    softmax_loss = softmax(con)
    est2 = softmax_loss.smooth_objective(observed, 'grad')
    np.testing.assert_allclose(est2, simple_estimator)

    loss = softmax_conjugate(con,
                             observed)

    f, coefs = loss._solve_conjugate_problem(simple_estimator)

    np.testing.assert_allclose(coefs, observed)

    np.testing.assert_allclose(loss.smooth_objective(simple_estimator, 'grad'), observed)
    loss.smooth_objective(coefs, 'both')

@set_seed_iftrue(SET_SEED)
def test_softmax_sigma_not1():

    sigma=2
    A = np.array([[-1.,0.]])
    b = np.array([-1.])

    con = constraints(A, b, covariance=sigma**2 * np.identity(2))
    observed = np.array([1.4,2.3])

    simple_estimator = con.estimate_mean(observed)
    softmax_loss = softmax(con,
                           sigma=sigma)
    est2 = softmax_loss.smooth_objective(observed, 'grad')*sigma**2
    np.testing.assert_allclose(est2, simple_estimator)

    loss = softmax_conjugate(con,
                             observed, 
                             sigma=sigma)


    loss.coefs[:] = 1.
    f, coefs = loss._solve_conjugate_problem(simple_estimator/sigma**2, niter=2000)

    np.testing.assert_allclose(coefs, observed)

    np.testing.assert_allclose(loss.smooth_objective(simple_estimator/sigma**2, 'grad'), observed)

    print(loss.smooth_objective(coefs, 'both')[0])
    print(est2, simple_estimator)

    G1 = softmax_loss.smooth_objective(observed, 'grad')
    np.testing.assert_allclose(loss.smooth_objective(G1, 'grad'), observed)

    G2 = loss.smooth_objective(G1, 'grad')
    np.testing.assert_allclose(softmax_loss.smooth_objective(G2, 'grad'), G1)

@set_seed_iftrue(SET_SEED)
def test_gaussian_unknown():

    n, p = 20, 5
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    T = X.T.dot(Y)
    N = -(Y**2).sum() / 2.

    sufficient_stat = np.hstack([T, N])

    cumulant = gaussian_cumulant(X)
    conj = gaussian_cumulant_conjugate(X)

    MLE = cumulant.regression_parameters(conj.smooth_objective(sufficient_stat, 'grad'))
    linear = rr.identity_quadratic(0, 0, -sufficient_stat, 0)
    cumulant.coefs[:] = 1.
    MLE2 = cumulant.solve(linear, tol=1.e-12, min_its=400)

    np.testing.assert_allclose(MLE2, conj.smooth_objective(sufficient_stat, 'grad'), rtol=1.e-4, atol=1.e-4)

    beta_hat = np.linalg.pinv(X).dot(Y)
    sigmasq_hat = np.sum(((Y - X.dot(beta_hat))**2) / n)

    np.testing.assert_allclose(beta_hat, MLE[0])
    np.testing.assert_allclose(sigmasq_hat, MLE[1])

    G = conj.smooth_objective(sufficient_stat, 'grad')
    M = cumulant.smooth_objective(G, 'grad')
    np.testing.assert_allclose(sufficient_stat, M)

    G = cumulant.smooth_objective(MLE2, 'grad')
    M = conj.smooth_objective(G, 'grad')
    np.testing.assert_allclose(MLE2, M)

@set_seed_iftrue(SET_SEED)
def test_gaussian_known():

    n, p = 20, 5
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    sigma = 2
    T = X.T.dot(Y)

    sufficient_stat = T

    cumulant = gaussian_cumulant_known(X, sigma)
    conj = gaussian_cumulant_conjugate_known(X, sigma)

    MLE = cumulant.regression_parameters(conj.smooth_objective(sufficient_stat, 'grad'))
    linear = rr.identity_quadratic(0, 0, -sufficient_stat, 0)
    cumulant.coefs[:] = 1.
    MLE2 = cumulant.solve(linear, tol=1.e-12, min_its=400)

    beta_hat = np.linalg.pinv(X).dot(Y)
    np.testing.assert_allclose(beta_hat, MLE)

    np.testing.assert_allclose(beta_hat / sigma**2, conj.smooth_objective(sufficient_stat, 'grad'))
    np.testing.assert_allclose(MLE2, conj.smooth_objective(sufficient_stat, 'grad'))

    G = conj.smooth_objective(sufficient_stat, 'grad')
    M = cumulant.smooth_objective(G, 'grad')
    np.testing.assert_allclose(sufficient_stat, M)

    G = cumulant.smooth_objective(sufficient_stat, 'grad')
    M = conj.smooth_objective(G, 'grad')
    np.testing.assert_allclose(sufficient_stat, M)

