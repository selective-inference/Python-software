import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm

class cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale, #equals the randomization variance in our case
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
        log_cube_grad = (np.true_divide(norm.cdf(arg_u) - norm.cdf(arg_l), cube_prob)/ self.randomization_scale).sum()

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
                 quadratic= None,
                 nstep = 10):

        self.t = t

        self.A = A

        self.target = target

        self.null_statistic = null_statistic

        E = active.sum()

        self.active = active

        self.randomization = randomizer

        self.nstep = nstep

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