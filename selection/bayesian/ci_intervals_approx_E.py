import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm

class log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange

        rr.smooth_atom.__init__(self,
                                (q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale

        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = np.log(cube_prob).sum()
        log_cube_grad = (np.true_divide(norm.pdf(arg_u) - norm.pdf(arg_l), cube_prob)/ self.randomization_scale).sum()

        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class approximate_conditional_prob(rr.smooth_atom):

    def __init__(self,
                 X,
                 target,
                 A, # the coef matrix of target
                 null_statistic, #null statistic that stays fixed
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 randomizer,
                 epsilon,
                 t, #point at which density is to computed
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t

        self.A = A

        self.target = target

        self.null_statistic = null_statistic

        E = active.sum()

        p = X.shape[1]

        self.q = p-E

        self.active = active

        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]
        self.active_lagrange = lagrange[active]

        #here, feasible point is in E dimensions
        initial = feasible_point

        self.feasible_point = feasible_point

        rr.smooth_atom.__init__(self,
                                (E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        nonnegative = nonnegative_softmax_scaled(E)

        X_E = self.X_E = X[:, active]
        self.X_inactive = X[:, ~active]

        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.B_active = (B_E + epsilon * np.identity(E)) * active_signs[None, :]
        self.B_inactive = B_mE * active_signs[None, :]

        self.subgrad_offset = active_signs * self.active_lagrange

        opt_vars = np.zeros(E, bool)
        opt_vars[:E] = 1

        self._opt_selector = rr.selector(opt_vars, (E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)

        self.E = E

    def sel_prob_smooth_objective(self, param, j, mode='both', check_feasibility=False):

        param = self.apply_offset(param)
        index = np.zeros(self.E, bool)
        index[j] = 1
        data = np.squeeze(self.t * self.A[:, index]) + self.A[:, ~index].dot(self.target[~index])

        offset_active = self.subgrad_offset + self.null_statistic[:self.E] + data[:self.E]

        offset_inactive = self.null_statistic[self.E:] + data[self.E:]

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.B_active, offset_active))

        cube_obj = log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = 1.)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.B_inactive, offset_inactive))

        total_loss = rr.smooth_sum([active_conj_loss,
                                    cube_loss,
                                    self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

