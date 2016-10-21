import numpy as np
import regreg.api as rr
from scipy.optimize import minimize
from selection.bayesian.dual_scipy import cube_barrier_softmax_coord, softmax_barrier_conjugate, log_barrier_conjugate, \
    dual_selection_probability_func
from selection.bayesian.dual_optimization import selection_probability_dual_objective
from selection.bayesian.selection_probability_rr import cube_subproblem, cube_gradient, cube_barrier, \
    selection_probability_objective
from selection.bayesian.selection_probability import selection_probability_methods
from selection.randomized.api import randomization
from selection.bayesian.credible_intervals import projected_langevin

class sel_prob_gradient_map(rr.smooth_atom):
    def __init__(self,
                 X,
                 primal_feasible,
                 dual_feasible,
                 active,
                 active_signs,
                 lagrange,
                 generative_X,
                 noise_variance,
                 randomizer,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.E = active.sum()
        self.n, self.p = X.shape
        self.dim = generative_X.shape[1]

        (self.X, self.primal_feasible, self.dual_feasible, self.active, self.active_signs, self.lagrange,
         self.generative_X, self.noise_variance, self.randomizer, self.epsilon) = \
            (X, primal_feasible, dual_feasible, active, active_signs, lagrange, generative_X, noise_variance,
             randomizer, epsilon)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):
        true_param = self.apply_offset(true_param)

        mean_parameter = np.squeeze(self.generative_X.dot(true_param))

        dual_sol = selection_probability_dual_objective(self.X,
                                                        self.dual_feasible,
                                                        self.active,
                                                        self.active_signs,
                                                        self.lagrange,
                                                        mean_parameter,
                                                        self.noise_variance,
                                                        self.randomizer,
                                                        self.epsilon)

        primal_sol = selection_probability_objective(self.X,
                                                     self.primal_feasible,
                                                     self.active,
                                                     self.active_signs,
                                                     self.lagrange,
                                                     mean_parameter,
                                                     self.noise_variance,
                                                     self.randomizer,
                                                     self.epsilon)

        if self.n + self.E > self.p:
            sel_prob_dual = dual_sol.minimize(max_its=1000, min_its=500, tol=1.e-12)[::-1]
            optimal_dual = mean_parameter - (dual_sol.X_permute.dot(np.linalg.inv(dual_sol.B_p.T))).\
                dot(sel_prob_dual[1])
            sel_prob_val = sel_prob_dual[0]
            optimizer = self.generative_X.T.dot(np.true_divide(optimal_dual - mean_parameter, self.noise_variance))

        else:
            sel_prob_primal = primal_sol.minimize(max_its=1000, min_its=500, tol=1.e-12)[::-1]
            optimal_primal = (sel_prob_primal[1])[:self.n]
            sel_prob_val = -sel_prob_primal[0]
            optimizer = self.generative_X.T.dot(np.true_divide(optimal_primal - mean_parameter, self.noise_variance))

        if mode == 'func':
            return sel_prob_val
        elif mode == 'grad':
            return optimizer
        elif mode == 'both':
            return sel_prob_val, optimizer
        else:
            raise ValueError('mode incorrectly specified')

class selective_map(rr.smooth_atom):
    def __init__(self,
                 y,
                 X,
                 primal_feasible,
                 dual_feasible,
                 active,
                 active_signs,
                 lagrange,
                 generative_X,
                 noise_variance,
                 prior_variance,
                 randomizer,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        self.param_shape = generative_X.shape[1]

        y = np.squeeze(y)

        initial = np.random.standard_normal(self.param_shape)

        self.generative_X = generative_X

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.set_likelihood(y, noise_variance, generative_X)

        self.set_prior(prior_variance)

        self.log_sel_prob = sel_prob_gradient_map(X,
                                                  primal_feasible,
                                                  dual_feasible,
                                                  active,
                                                  active_signs,
                                                  lagrange,
                                                  generative_X,
                                                  noise_variance,
                                                  randomizer,
                                                  epsilon)

        self.total_loss = rr.smooth_sum([self.likelihood_loss,
                                         self.log_prior_loss,
                                         self.log_sel_prob])

    def set_likelihood(self, y, noise_variance, generative_X):
        likelihood_loss = rr.signal_approximator(y, coef=1. / noise_variance)
        self.likelihood_loss = rr.affine_smooth(likelihood_loss, generative_X)

    def set_prior(self, prior_variance):
        self.log_prior_loss = rr.signal_approximator(np.zeros(self.param_shape), coef=1. / prior_variance)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False):

        true_param = self.apply_offset(true_param)

        if mode == 'func':
            f = self.total_loss.smooth_objective(true_param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss.smooth_objective(true_param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = self.total_loss.smooth_objective(true_param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")













