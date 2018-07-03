import numpy as np
import functools

from ...tests.decorators import set_seed_iftrue
from ..selective_MLE_utils import barrier_solve_, barrier_solve_affine_

from .test_selective_MLE_onedim import solve_barrier_nonneg

@set_seed_iftrue(True)
def test_C_solver():

    X = np.random.standard_normal((10, 5))
    precision = X.T.dot(X) / 10
    conjugate_arg = np.random.standard_normal(5)

    soln1, val1, hess1 = solve_barrier_nonneg(conjugate_arg,
                                              precision,
                                              tol=1.e-12)

    grad, opt_val, opt_proposed = np.ones((3, 5))
    scaling = np.sqrt(np.diag(precision))

    val2, soln2, hess2 = barrier_solve_(grad,
                                        opt_val,
                                        opt_proposed,
                                        conjugate_arg,
                                        precision,
                                        scaling,
                                        1.,
                                        value_tol=1.e-12)

    np.testing.assert_allclose(soln1, soln2, atol=1.e-4, rtol=1.e-4)
    np.testing.assert_allclose(hess1, hess2, atol=1.e-4, rtol=1.e-4)
    assert (np.fabs(val1 - val2) < 1.e-4 * np.fabs(val1))

@set_seed_iftrue(True)
def test_affine_solver():

    X = np.random.standard_normal((10, 5))
    precision = X.T.dot(X) / 10
    conjugate_arg = np.random.standard_normal(5)


    grad, opt_val, opt_proposed = np.ones((3, 5))
    scaling = np.sqrt(np.diag(precision))

    val1, soln1, hess1 = barrier_solve_(grad,
                                        opt_val,
                                        opt_proposed,
                                        conjugate_arg,
                                        precision,
                                        scaling,
                                        1.,
                                        value_tol=1.e-12)

    grad, opt_val, opt_proposed = np.ones((3, 5))
    scaling = np.sqrt(np.diag(precision))

    A = -np.identity(5)
    b = np.zeros(5),
    val2, soln2, hess2 = barrier_solve_affine_(grad,
                                               opt_val,
                                               opt_proposed,
                                               conjugate_arg,
                                               precision,
                                               scaling,
                                               A,
                                               np.zeros(5),
                                               -1 * opt_val,
                                               1.,
                                               value_tol=1.e-12)

    np.testing.assert_allclose(soln1, soln2, atol=1.e-4, rtol=1.e-4)
    print(soln1)
    print(soln2)
    
    print(val1)
    print(val2)
    assert (np.fabs(val1 - val2) < 1.e-4 * np.fabs(val1))

    print(hess1, hess1.shape)
    print(hess2, hess2.shape)
    np.testing.assert_allclose(hess1, hess2, atol=1.e-4, rtol=1.e-4)


