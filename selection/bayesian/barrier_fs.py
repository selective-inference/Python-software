import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import cube_barrier_scaled, cube_gradient_scaled, cube_hessian_scaled
from selection.algorithms.softmax import nonnegative_softmax

class barrier_conjugate_fs_rr(rr.smooth_atom):
    """

    Conjugate of a barrier for the
    product $[0,\infty)^E \times [-\lambda,\lambda]^{-E}$.
    """

    def __init__(self,
                 cube_bool,  # -E
                 cube_scale = 1.,
                 barrier_scale=1.,  # maybe scale each coordinate in future?
                 coef=1.,
                 offset=None,
                 quadratic=None):

        p = cube_bool.shape[0]
        orthant_bool = ~cube_bool

        initial = -np.ones(p)
        self._initial = initial[cube_bool] = 0.5 * np.ones(p-1)

        if barrier_scale is None:
            barrier_scale = 1.

        (self.cube_bool,
         self.orthant_bool,
         self.cube_scale,
         self.barrier_scale) = (cube_bool,
                                orthant_bool,
                                cube_scale,
                                barrier_scale)

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-12):

        # here we compute those expressions in the note

        arg = self.apply_offset(arg)  # all smooth_objectives should do this....

        cube_arg = arg[self.cube_bool]

        orthant_arg = arg[self.orthant_bool]

        if check_feasibility and np.any(orthant_arg >= -tol):
            raise ValueError('returning nan gradient')
            if mode == 'func':
                return np.inf
            elif mode == 'grad':
                return np.nan * np.ones(self.shape)
            elif mode == 'both':
                return np.inf, np.nan * np.ones(self.shape)
            else:
                raise ValueError('mode incorrectly specified')

        maximizer, neg_val = fs_conjugate(np.append(orthant_arg, cube_arg))

        if np.any(np.isnan(maximizer)):
            raise ValueError('cube maximizer is nan')

        val = -neg_val

        if mode == 'func':
            return val
        elif mode == 'grad':
            return maximizer
        elif mode == 'both':
            return val, maximizer
        else:
            raise ValueError('mode incorrectly specified')


class linear_map(rr.smooth_atom):
    def __init__(self,
                 dual_arg,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.dual_arg = dual_arg
        p = self.dual_arg.shape[0]
        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):
        arg = self.apply_offset(arg)

        if mode == 'func':
            f = self.dual_arg.T.dot(arg)
            return f
        elif mode == 'grad':
            g = self.dual_arg
            return g
        elif mode == 'both':
            f = self.dual_arg.T.dot(arg)
            g = self.dual_arg
            return f, g
        else:
            raise ValueError('mode incorrectly specified')

def fs_conjugate(argument,
                 nstep=30,
                 initial=None,
                 lipschitz=0,
                 tol=1.e-10):

    k = argument.shape[0]
    E = 1
    if initial is None:
        current = np.append(1, 0.5* np.ones(k-1))
    else:
        current = initial

    current_value = np.inf

    step = np.ones(k, np.float)

    linear = linear_map(argument)

    nonnegative = nonnegative_softmax(E)

    objective = lambda z: cube_barrier_scaled(z[E:], z[:E] * np.ones(k - 1))+ nonnegative.smooth_objective(z[:E],'func')\
                          - linear.smooth_objective(z, 'func')

    arg_active = argument[:E]

    arg_inactive = argument[E:]

    for itercount in range(nstep):
        #print "loc", current
        current_active = current[:E]
        current_inactive = current[E:]

        lagrange = current_active * np.ones(k - 1)

        diff_1 = (4 * (lagrange ** 2)) - (current_inactive ** 2)
        diff_2 = (lagrange ** 2) - (current_inactive ** 2)

        inter_grad = 6. * (np.true_divide(current_inactive ** 2, diff_1 * diff_2).sum())

        inter_hessian = (-64./(diff_1**2) + 4./ (diff_2**2)).sum()

        gradient_active = arg_active + 1./(current_active * (current_active+1)) \
                          + current_active * inter_grad

        gradient_inactive = arg_inactive + cube_gradient_scaled(current_inactive, lagrange)

        hessian_inactive = cube_hessian_scaled(current_inactive, lagrange)
        hessian_active = -1./(current_active**2) + 1./(current_active**2 + 1.) + inter_grad -\
                         ((current_active**2)* inter_hessian)

        gradient_obj = np.append(gradient_active, gradient_inactive)

        hessian_obj = np.append(hessian_active, hessian_inactive)

        newton_step =np.true_divide(gradient_obj, hessian_obj+lipschitz)

        #print "loc",diff_1, diff_2, gradient_obj, hessian_obj

        # make sure proposal is a descent

        count = 0
        while True:
            proposal = current - step * newton_step
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5

        # stop if relative decrease is small
        #print "loc-1", current, np.fabs(current_value - proposed_value)

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