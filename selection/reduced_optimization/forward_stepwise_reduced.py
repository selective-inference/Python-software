from math import log
import numpy as np
import regreg.api as rr
from scipy.stats import norm

class neg_log_cube_probability_fs(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, par, mode='both', check_feasibility=False, tol=1.e-6):

        par = self.apply_offset(par)

        arg = par[0]
        mu = par[0:]

        arg_u = ((arg *np.ones(self.q)) + mu) / self.randomization_scale
        arg_l = (-(arg *np.ones(self.q)) + mu) / self.randomization_scale
        prod_arg = np.exp(-(2. * self.mu * (arg *np.ones(self.q))) / (self.randomization_scale ** 2))
        neg_prod_arg = np.exp((2. * self.mu * (arg *np.ones(self.q))) / (self.randomization_scale ** 2))

        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()

        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(self.mu > 0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)

        log_cube_grad_vec_arg = np.zeros(self.q)
        log_cube_grad_vec_arg[indicator] = -(np.true_divide(norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                                    cube_prob[indicator])) / self.randomization_scale

        log_cube_grad_vec_arg[pos_index] = ((1. + prod_arg[pos_index]) /
                                    ((prod_arg[pos_index] / arg_u[pos_index]) -
                                     (1. / arg_l[pos_index]))) / (self.randomization_scale ** 2)

        log_cube_grad_vec_arg[neg_index] = ((arg_u[neg_index] - (arg_l[neg_index] * neg_prod_arg[neg_index]))
                                    / (self.randomization_scale ** 2)) / (1. + neg_prod_arg[neg_index])

        log_cube_grad_arg = log_cube_grad_vec_arg.sum()

        log_cube_grad_vec_mu = np.zeros(self.q)
        log_cube_grad_vec_mu[indicator] = -(np.true_divide(norm.pdf(arg_u[indicator]) - norm.pdf(arg_l[indicator]),
                                                            cube_prob[indicator])) / self.randomization_scale
        log_cube_grad_vec_mu[pos_index] = ((1. - prod_arg[pos_index]) /
                                            (-(prod_arg[pos_index] / arg_u[pos_index]) +
                                             (1. / arg_l[pos_index]))) / (self.randomization_scale ** 2)
        log_cube_grad_vec_mu[neg_index] = ((arg_u[neg_index] - (arg_l[neg_index] * neg_prod_arg[neg_index]))
                                            / (self.randomization_scale ** 2)) / (1. - neg_prod_arg[neg_index])

        log_cube_grad = np.append(log_cube_grad_vec_mu, log_cube_grad_arg)


        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")