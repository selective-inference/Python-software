import numpy as np
import nose.tools as nt
import numpy.testing.decorators as dec

from ...tests.instance import gaussian_instance as instance

from ..lasso import lasso
from ..debiased_lasso import (debiased_lasso_inference,
                              _find_row_approx_inverse_X,
                              debiasing_matrix)

# for regreg implementation comparison

from regreg.api import (quadratic_loss,
                        identity_quadratic,
                        l1norm,
                        simple_problem)

try:
    import rpy2.robjects as rpy
    rpy2_available = True
    import rpy2.robjects.numpy2ri as numpy2ri
    rpy.r('library(selectiveInference)')
except ImportError:
    rpy2_available = False


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
    X = np.random.standard_normal((n, p))
    j = 5
    delta = 0.30

    X[:, 3] = X[:, 3] + X[:, j]
    X[:, 10] = X[:, 10] + X[:, j]
    S = X.T.dot(X) / n

    soln = _find_row_approx_inverse(S, j, delta, solve_args={'min_its': 500, 'tol': 1.e-14, 'max_its': 1000})

    soln_C = _find_row_approx_inverse_X(X, j, delta, kkt_tol=1.e-14, parameter_tol=1.e-14, maxiter=1000,
                                        objective_tol=1.e-14)
    soln_C2 = debiasing_matrix(X, j, delta, kkt_tol=1.e-14, parameter_tol=1.e-14, max_iter=1000, objective_tol=1.e-14,
                               linesearch=False)

    # make sure linesearch terminates

    debiasing_matrix(X, j, delta, linesearch=True)

    basis_vector = np.zeros(p)
    basis_vector[j] = 1.

    nt.assert_true(np.fabs(S.dot(soln) - basis_vector).max() < delta * 1.001)

    U = - S.dot(-soln) - basis_vector

    yield np.testing.assert_allclose, soln_C, soln_C2
    yield nt.assert_true, np.fabs(U).max() < delta * 1.001
    yield nt.assert_equal, np.sign(U[j]), -np.sign(soln[j])
    yield nt.assert_raises, ValueError, _find_row_approx_inverse, S, j, 1.e-7 * delta
    yield np.testing.assert_allclose, soln, soln_C, 1.e-3, 1.e-3


def test_approx_inverse_nondegen(n=100, p=20):
    X = np.random.standard_normal((n, p))
    j = 5
    delta = 0.30

    X[:, 3] = X[:, 3] + X[:, j]
    X[:, 10] = X[:, 10] + X[:, j]

    M = debiasing_matrix(X, np.arange(p))

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_compareR(n=100, p=30):
    X = np.random.standard_normal((n, p))
    j = 5
    delta = 0.30

    X[:, 3] = X[:, 3] + X[:, j]
    X[:, 10] = X[:, 10] + X[:, j]
    S = X.T.dot(X) / n

    numpy2ri.activate()
    rpy.r.assign('X', X)
    rpy.r.assign('j', j + 1)
    rpy.r('soln = selectiveInference:::debiasingMatrix(X, TRUE, nrow(X), j)')
    soln_R = np.squeeze(np.asarray(rpy.r('soln')))

    soln_py = debiasing_matrix(X, j)

    yield np.testing.assert_allclose, soln_R, soln_py
    
    j = np.array([3,5])
    numpy2ri.activate()
    rpy.r.assign('X', X)
    rpy.r.assign('j', j+1)
    rpy.r('soln = selectiveInference:::debiasingMatrix(X, TRUE, nrow(X), j)')
    soln_R = np.squeeze(np.asarray(rpy.r('soln')))

    soln_py = debiasing_matrix(X, j)

    yield np.testing.assert_allclose, soln_R, soln_py
    
    j = np.array([3, 5])
    numpy2ri.activate()
    rpy.r.assign('X', X)
    rpy.r.assign('j', j + 1)
    rpy.r('soln = selectiveInference:::debiasingMatrix(X, TRUE, nrow(X), j)')
    soln_R = np.squeeze(np.asarray(rpy.r('soln')))

    soln_py = debiasing_matrix(X, j)

    yield np.testing.assert_allclose, soln_R, soln_py
    
    numpy2ri.deactivate()


## regreg implementation

def _find_row_approx_inverse(Sigma, j, delta, solve_args={'min_its': 100, 'tol': 1.e-6, 'max_its': 500}):
    """
    Find an approximation of j-th row of inverse of Sigma.
    Solves the problem
    .. math::
        \text{min}_{\theta} \frac{1}{2} \theta^TS\theta
    subject to $\|\Sigma \hat{\theta} - e_j\|_{\infty} \leq \delta$ with
    $e_j$ the $j$-th elementary basis vector and `S` as $\Sigma$,
    and `delta` as $\delta$.
    Described in Table 1, display (4) of https://arxiv.org/pdf/1306.3171.pdf
    """
    p = Sigma.shape[0]
    elem_basis = np.zeros(p, np.float)
    elem_basis[j] = 1.
    loss = quadratic_loss(p, Q=Sigma)
    penalty = l1norm(p, lagrange=delta)
    iq = identity_quadratic(0, 0, elem_basis, 0)
    problem = simple_problem(loss, penalty)
    dual_soln = problem.solve(iq, **solve_args)

    soln = -dual_soln

    # check feasibility -- if it fails miserably
    # presume delta was too small

    feasibility_gap = np.fabs(Sigma.dot(soln) - elem_basis).max()
    if feasibility_gap > (1.01) * delta:
        raise ValueError('does not seem to be a feasible point -- try increasing delta')

    return soln

