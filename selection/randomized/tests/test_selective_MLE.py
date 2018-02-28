import numpy as np
import functools

from ...tests.decorators import set_seed_iftrue
from ..selective_MLE_utils import barrier_solve_

def solve_barrier_nonneg(conjugate_arg,
                         precision,
                         feasible_point=None,
                         step=1,
                         nstep=150,
                         tol=1.e-8):

    scaling = np.sqrt(np.diag(precision))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. + np.log(1.+ 1./(u / scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) + (1./(scaling + u) - 1./u)
    barrier_hessian = lambda u: (-1./((scaling + u)**2.) + 1./(u**2.))

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        newton_step = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * newton_step
            if np.all(proposal > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            proposal = current - step * newton_step
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + np.diag(barrier_hessian(current)))
    return current, current_value, hess

@set_seed_iftrue(True)
def test_C_solver():

    X = np.random.standard_normal((10, 5))
    precision = X.T.dot(X) / 10
    conjugate_arg = np.random.standard_normal(5)


    soln1, val1, _ = solve_barrier_nonneg(conjugate_arg,
                                          precision,
                                          tol=1.e-12)

    grad, opt_val, opt_proposed = np.ones((3, 5))
    scaling = np.sqrt(np.diag(precision))

    soln2, val2 = barrier_solve_(grad,
                                 opt_val,
                                 opt_proposed,
                                 conjugate_arg,
                                 precision,
                                 scaling,
                                 value_tol=1.e-12)

    np.testing.assert_allclose(soln1, soln2, atol=1.e-4, rtol=1.e-4)
    assert (np.fabs(val1 - val2) < 1.e-4 * np.fabs(val1))

