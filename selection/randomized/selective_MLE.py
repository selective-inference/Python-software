import numpy as np
import functools

def solve_UMVU(target_transform,
               opt_transform,
               target_observed,
               feasible_point,
               target_cov,
               randomizer_precision):

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

    #print("shapes", target_precision.dot(sel_MLE).shape,  A.T.dot(randomizer_precision).shape, offset_term.shape)
    #implied_parameter = np.hstack([target_precision.dot(sel_MLE)-A.T.dot(randomizer_precision).dot(conditioned_value),
    #                               offset_term*np.ones((1,1))])

    print("selective MLE", sel_MLE)
    return np.squeeze(sel_MLE)
        #, inv_hessian, mle_partial, implied_cov, implied_cov.dot(implied_parameter), mle_transform

def solve_barrier_nonneg(conjugate_arg,
                         precision,
                         feasible_point=None,
                         step=1,
                         nstep=1000,
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