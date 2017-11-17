import numpy as np
import nose.tools as nt
import numpy.testing.decorators as dec

from selection.tests.instance import gaussian_instance as instance
import selection.tests.reports as reports

from selection.algorithms.lasso import lasso 
from selection.algorithms.debiased_lasso import (debiased_lasso_inference,
                                                 _find_row_approx_inverse,
                                                 _find_row_approx_inverse_X)
import regreg.api as rr

def test_gaussian(n=100, p=20):

    X, y, beta = instance(n=n, p=p, sigma=1.)[:3]

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))

    weights = 1.1 * lam_theor * np.ones(p)
    weights[:3] = 0.

    L = lasso.gaussian(X, y, weights, sigma=1.)
    L.ignore_inactive_constraints = True
    L.fit()

    print(debiased_lasso_inference(L, L.active, np.sqrt(2 * np.log(p) / n)))
    print(beta)

def test_approx_inverse(n=50, p=100):

    n, p = 50, 100
    X = np.random.standard_normal((n, p))
    j = 5
    delta = 0.30
    
    X[:,3] = X[:,3] + X[:,j]
    X[:,10] = X[:,10] + X[:,j]
    S = X.T.dot(X) / n
    
    soln = _find_row_approx_inverse(S, j, delta, solve_args={'min_its':500, 'tol':1.e-14, 'max_its':1000} )

    soln_C = _find_row_approx_inverse_X(X, j, delta, kkt_tol=1.e-14, parameter_tol=1.e-14, maxiter=1000, objective_tol=1.e-14)

    basis_vector = np.zeros(p)
    basis_vector[j] = 1.

    nt.assert_true(np.fabs(S.dot(soln) - basis_vector).max() < delta * 1.001)

    U = - S.dot(-soln) - basis_vector

    yield nt.assert_true, np.fabs(U).max() < delta * 1.001
    yield nt.assert_equal, np.sign(U[j]), -np.sign(soln[j])
    yield nt.assert_raises, ValueError, _find_row_approx_inverse, S, j, 1.e-7 * delta
    yield np.testing.assert_allclose, soln, soln_C, 1.e-3
