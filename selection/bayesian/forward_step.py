import numpy as np

from selection.algorithms.softmax import nonnegative_softmax
import regreg.api as rr
from selection.bayesian.selection_probability_rr import cube_barrier_scaled, cube_gradient_scaled, cube_hessian_scaled

def cube_subproblem_fs(argument,
                           c,
                           randomization_CGF_conjugate,
                           lagrange= 1., nstep=100,
                           initial=None,
                           lipschitz=0,
                           tol=1.e-10):
    '''
    Solve the subproblem
    $$
    \text{minimize}_{z} \Lambda_{-E}^*(u + c.z_{-E}) + b_{-E}(z)
    $$
    where $u$ is `argument`, $\Lambda_{-E}^*$ is the
    conjvex conjugate of the $-E$ coordinates of the
    randomization (assumes that randomization has independent
    coordinates) and
    $b_{-E}$ is a barrier approximation to
    the cube $\prod_{j \in -E} [-\lambda_j,\lambda_j]$ with
    $\lambda$ being `lagrange`.
    Returns the maximizer and the value of the convex conjugate.
    '''
    k = argument.shape[0]
    if initial is None:
        current = np.zeros(k, np.float)
    else:
        current = initial  # no copy

    current_value = np.inf

    conj_value = lambda x: randomization_CGF_conjugate.smooth_objective(x, 'func')
    conj_grad = lambda x: randomization_CGF_conjugate.smooth_objective(x, 'grad')

    step = np.ones(k, np.float)
    objective = lambda u: cube_barrier_scaled(u, lagrange) + conj_value(argument + c*u)

    for itercount in range(nstep):
        newton_step = ((cube_gradient_scaled(current, lagrange) +
                        (c*conj_grad(argument + c*current))) /
                       (cube_hessian_scaled(current, lagrange) + lipschitz))

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * newton_step
            failing = (proposal > lagrange) + (proposal < - lagrange)
            if not failing.sum():
                break
            step *= 0.5 ** failing

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

    value = objective(current)
    return current, value

