import numpy as np
import nose.tools as nt
import numpy.testing.decorators as dec

from selection.tests.instance import gaussian_instance as instance
import selection.tests.reports as reports

from selection.algorithms.lasso import lasso 
from selection.algorithms.debiased_lasso import (debiased_lasso_inference,
                                                 _find_row_approx_inverse)
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

def test_approx_inverse():

    n, p = 50, 100
    X = np.random.standard_normal((n, p))
    S = X.T.dot(X) / n
    j = 5
    delta = 0.60
    
    soln = _find_row_approx_inverse(S, j, delta)

    basis_vector = np.zeros(p)
    basis_vector[j] = 1.

    nt.assert_true(np.fabs(S.dot(soln) - basis_vector).max() < delta * 1.001)

    U = - S.dot(-soln) - basis_vector
    nt.assert_true(np.fabs(U).max() < delta * 1.001)
    nt.assert_equal(np.argmax(np.fabs(U)), j)
    nt.assert_equal(np.sign(U[j]), -np.sign(soln[j]))
    nt.assert_raises(ValueError, _find_row_approx_inverse, S, j, 1.e-7 * delta)
