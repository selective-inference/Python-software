from itertools import product
import numpy as np
import nose.tools as nt

from ..convenience import lasso, step, threshold

def test_marginalize():

    np.random.seed(10) # we are going to freeze the active set for this test

    n, p = 20, 5
    X = np.random.standard_normal((n, p))
    X /= np.sqrt((X**2).sum(0))[None, :]
    Y = X.dot([60.1, -61, 0, 0, 0]) + np.random.standard_normal(n)

    n, p = X.shape

    W = np.ones(p) * 20
    L = lasso.gaussian(X, Y, W, randomizer='gaussian', randomizer_scale=0.01)
    signs = L.fit()

    # we should be able to reconstruct the initial randomness by hand

    beta = L._view.initial_soln
    omega = X.T.dot(X.dot(beta) - Y) + L.ridge_term * beta + L._view.initial_subgrad

    np.testing.assert_allclose(omega, L._view._initial_omega)

    A1, b1 = L._view.opt_transform
    opt_state1 = L._view.observed_opt_state.copy()
    state1 = A1.dot(opt_state1) + b1

    # now marginalize over some coordinates of inactive

    marginalizing_groups = np.ones(p, np.bool)
    marginalizing_groups[:3] = False

    L.decompose_subgradient(marginalizing_groups = marginalizing_groups)

def test_condition():

    n, p = 20, 5

    np.random.seed(10) # we are going to freeze the active set for this test

    X = np.random.standard_normal((n, p))
    X /= np.sqrt((X**2).sum(0))[None, :]
    Y = X.dot([60.1, -61, 0, 0, 0]) + np.random.standard_normal(n)

    n, p = X.shape

    W = np.ones(p) * 20
    L = lasso.gaussian(X, Y, W, randomizer='gaussian', randomizer_scale=0.01)

    signs = L.fit()

    # we should be able to reconstruct the initial randomness by hand

    beta = L._view.initial_soln
    omega = X.T.dot(X.dot(beta) - Y) + L.ridge_term * beta + L._view.initial_subgrad

    np.testing.assert_allclose(omega, L._view._initial_omega)

    A1, b1 = L._view.opt_transform
    state1 = A1.dot(L._view.observed_opt_state) + b1

    # now marginalize over some coordinates of inactive

    conditioning_groups = np.ones(p, np.bool)
    conditioning_groups[:3] = False

    L.decompose_subgradient(conditioning_groups = conditioning_groups)

def test_both():


    np.random.seed(10) # we are going to freeze the active set for this test

    n, p = 20, 10
    X = np.random.standard_normal((n, p))
    X /= np.sqrt((X**2).sum(0))[None, :]
    Y = X.dot([60.1, -61] + [0] * (p-2)) + np.random.standard_normal(n)

    n, p = X.shape

    W = np.ones(p) * 20
    L = lasso.gaussian(X, Y, W, randomizer='gaussian', randomizer_scale=0.01)
    signs = L.fit()

    # we should be able to reconstruct the initial randomness by hand

    beta = L._view.initial_soln
    omega = X.T.dot(X.dot(beta) - Y) + L.ridge_term * beta + L._view.initial_subgrad

    np.testing.assert_allclose(omega, L._view._initial_omega)

    A1, b1 = L._view.opt_transform
    opt_state1 = L._view.observed_opt_state.copy()
    state1 = A1.dot(opt_state1) + b1

    # now marginalize over some coordinates of inactive

    marginalizing_groups = np.zeros(p, np.bool)
    marginalizing_groups[3:5] = True

    conditioning_groups = np.zeros(p, np.bool)
    conditioning_groups[5:7] = True

    L.decompose_subgradient(marginalizing_groups = marginalizing_groups,
                            conditioning_groups = conditioning_groups)

