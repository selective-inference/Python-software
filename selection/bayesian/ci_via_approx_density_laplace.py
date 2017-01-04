import time
import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov


class neg_log_cube_probability_laplace(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.b = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.b
        arg_l = (arg - self.lagrange)/self.b
        scaled_lagrange = (2* self.lagrange)/self.b

        ind_arg_1 = np.zeros(self.q, bool)
        ind_arg_1[(arg_u <0.)] = 1
        ind_arg_2 = np.zeros(self.q, bool)
        ind_arg_2[(arg_l >0.)] = 1
        ind_arg_3 = np.logical_and(~ind_arg_1, ind_arg_2)
        cube_prob = np.zeros(self.q)
        cube_prob[ind_arg_1] = np.exp(arg_u[ind_arg_1])/2. - np.exp(arg_l[ind_arg_1])/2.
        cube_prob[ind_arg_2] = -np.exp(-arg_u[ind_arg_2])/2. + np.exp(-arg_l[ind_arg_2])/2.
        cube_prob[ind_arg_3] = 1- np.exp(-arg_u[ind_arg_3])/2. - np.exp(arg_l[ind_arg_3])/2.
        log_cube_prob = -np.log(cube_prob).sum()

        log_cube_grad = np.zeros(self.q)
        log_cube_grad[ind_arg_1] = 1./self.b
        log_cube_grad[ind_arg_2] = np.true_divide((np.exp(-scaled_lagrange[ind_arg_2])-1.)/self.b,
                                                  1. - np.exp(-scaled_lagrange[ind_arg_2]))
        num_cube_grad = np.true_divide(np.exp(-scaled_lagrange[ind_arg_3]), 2 * self.b) - \
                        np.true_divide(np.exp((2* arg_l[ind_arg_3])), 2 * self.b)
        den_cube_grad = np.exp(arg_l[ind_arg_3]) - np.exp(-scaled_lagrange[ind_arg_3])/2. - \
                        np.true_divide(np.exp((2* arg_l[ind_arg_3])), 2)
        log_cube_grad[ind_arg_3] = num_cube_grad/den_cube_grad

        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")






