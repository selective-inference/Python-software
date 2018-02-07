from functools import partial

import numpy as np

from regreg.api import power_L

from .selective_MLE_utils import barrier_solve_

def solve_barrier_nonneg(conjugate_arg,
                         precision,
                         initial=None,
                         step=None,
                         max_iter=150,
                         value_tol=1.e-6):
    """
    Solve a smoothed version of the problem

    .. math::
    
        \text{minimize}_{\beta \geq 0} -u^T\beta + \frac{1}{2} \beta^T\Theta \beta

    with `conjugate_arg` as $u$ and `precision` as $\Theta$. The smoothing
    is done by adding a barrier function with scale determined
    by the diagonal of precision.

    Parameters
    ----------

    conjugate_arg: np.float(p)
        The value of the problem is a convex conjugate -- this is the
        argument to that function.

    precision: np.float((p,p))
        A non-negative definite matrix -- precision meaning the inverse
        of a covariance matrix.

    initial: np.float(p)
        Optional warm start.

    step: float
        An initial step size. Defaults to inverse of
        (approximate) largest eigenvalue of precision.

    max_iter: int
        When to stop optimization.

    value_tol: float
        Relative decrease in value for stopping.
    
    Returns
    -------

    value: float
        The value of the optimization problem.

    soln: np.float(p)
        The solution to the optimization problem,
        also the gradient of the value function.

    hess: np.float(p)
        The Hessian of the value function.

    """

    p = precision.shape[0]
    scaling = np.sqrt(np.diag(precision))

    if initial is None:
        initial, proposed, grad = np.zeros((3, p))

    if step is None:
        step = 1. / power_L(precision)

    soln, val = barrier_solve_(grad,
                               initial,
                               proposed,
                               conjugate_arg,
                               precision,
                               scaling,
                               step,
                               value_tol=value_tol)

    barrier_hessian = lambda u: (-1./((scaling + u)**2.) + 1./(u**2.))
    hess = np.linalg.inv(precision + np.diag(barrier_hessian(soln)))

    return val, soln, hess

def selective_MLE(target_observed,
                  target_cov,
                  target_transform,
                  opt_transform,
                  feasible_point,
                  randomizer_precision,
                  step=1,
                  max_iter=30,
                  tol=1.e-8):

    """

    Parameters
    ----------

    target_observed: np.float
        The observed value of our target estimator.
    
    target_cov: np.float
        Covariance matrix of target estimator.

    target_transform: tuple
        A pair (A, b) consisting of a linear transformation A and an offset b
        representing an affine transformation $x \mapsto Ax+b$.
        This transform should be computed as part of a linear decomposition of the
        score of an optimization problem with respect to a target
        of interest.

    opt_transform: tuple
        A pair (A, b) consisting of a linear transformation A and an offset b
        representing an affine transformation $x \mapsto Ax+b$.
        This transformation usually comes from the KKT conditions
        of an appropriate (randomized) optimization problem.

    feasible_point: np.float
        An appropriate feasible point for the optimization
        problem in the approximate likelihood.

    randomization_precision: np.float((p,p))
        Precision matrix of randomization in the randomized
        optimization problem.

    step: float
        An initial step size. Defaults to inverse of
        (approximate) largest eigenvalue of precision.

    max_iter: int
        When to stop optimization.

    value_tol: float
        Relative decrease in value for stopping.
    
    
    Returns
    -------

    XXXX

    """

    A, data_offset = target_transform # data_offset = N
    B, opt_offset = opt_transform     # opt_offset = u

    nopt = B.shape[1]
    ntarget = A.shape[1]

    # setup joint implied covariance matrix

    target_precision = np.linalg.inv(target_cov)

    implied_precision = np.zeros((ntarget + nopt, ntarget + nopt))
    implied_precision[:ntarget,:ntarget] = A.T.dot(randomizer_precision).dot(A) + target_precision
    implied_precision[:ntarget,ntarget:] = A.T.dot(randomizer_precision).dot(B)
    implied_precision[ntarget:,:ntarget] = B.T.dot(randomizer_precision).dot(A)
    implied_precision[ntarget:,ntarget:] = B.T.dot(randomizer_precision).dot(B)
    implied_cov = np.linalg.inv(implied_precision)

    implied_opt = implied_cov[ntarget:,ntarget:]
    implied_target = implied_cov[:ntarget,:ntarget]
    implied_cross = implied_cov[:ntarget,ntarget:]

    L = implied_cross.dot(np.linalg.inv(implied_opt))
    M_1 = np.linalg.inv(implied_precision[:ntarget,:ntarget]).dot(target_precision)
    M_2 = -np.linalg.inv(implied_precision[:ntarget,:ntarget]).dot(A.T.dot(randomizer_precision))

    conditioned_value = data_offset + opt_offset

    linear_term = implied_precision[ntarget:,ntarget:].dot(implied_cross.T.dot(np.linalg.inv(implied_target)))
    offset_term = -B.T.dot(randomizer_precision).dot(conditioned_value)

    natparam_transform = (linear_term, offset_term)
    conditional_natural_parameter = linear_term.dot(target_observed) + offset_term

    conditional_precision = implied_precision[ntarget:,ntarget:]

    M_1_inv = np.linalg.inv(M_1)
    mle_offset_term = - M_1_inv.dot(M_2.dot(conditioned_value))
    mle_transform = (M_1_inv, -M_1_inv.dot(L), mle_offset_term)
    var_transform = (-implied_precision[ntarget:,:ntarget].dot(M_1),
                     -implied_precision[ntarget:,:ntarget].dot(M_2.dot(conditioned_value)))

    cross_covariance = np.linalg.inv(implied_precision[:ntarget, :ntarget]).dot(implied_precision[:ntarget, ntarget:])
    var_matrices = (np.linalg.inv(implied_opt), np.linalg.inv(implied_precision[:ntarget,:ntarget]),
                    cross_covariance,target_precision)

    def mle_map(natparam_transform, mle_transform, var_transform, var_matrices,
                feasible_point, conditional_precision, target_observed):

        param_lin, param_offset = natparam_transform
        mle_target_lin, mle_soln_lin, mle_offset = mle_transform

        soln, value, _ = solve_barrier_nonneg(param_lin.dot(target_observed) + param_offset,
                                              conditional_precision,
                                              feasible_point=feasible_point,
                                              step=1,
                                              nstep=2000,
                                              tol=1.e-8)

        selective_MLE = mle_target_lin.dot(target_observed) + mle_soln_lin.dot(soln) + mle_offset

        var_target_lin, var_offset = var_transform
        var_precision, inv_precision_target, cross_covariance, target_precision =  var_matrices
        _, _, hess = solve_barrier_nonneg(var_target_lin.dot(selective_MLE) + var_offset + mle_offset,
                                          var_precision,
                                          feasible_point=None,
                                          step=1,
                                          nstep=2000)

        hessian = target_precision.dot(inv_precision_target +
                                       cross_covariance.dot(hess).dot(cross_covariance.T)).dot(target_precision)

        return selective_MLE, np.linalg.inv(hessian)

    mle_partial = functools.partial(mle_map, natparam_transform, mle_transform, var_transform, var_matrices,
                                    feasible_point, conditional_precision)
    sel_MLE, inv_hessian = mle_partial(target_observed)

    implied_parameter = np.hstack([target_precision.dot(sel_MLE)-A.T.dot(randomizer_precision).dot(conditioned_value), offset_term])

    return np.squeeze(sel_MLE), inv_hessian, mle_partial, implied_cov, implied_cov.dot(implied_parameter), mle_transform
