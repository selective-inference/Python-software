import numpy as np
from scipy.optimize import minimize
from selection.bayesian.dual_scipy import cube_barrier_softmax_coord, softmax_barrier_conjugate, log_barrier_conjugate, \
    dual_selection_probability_func
from selection.bayesian.dual_optimization import selection_probability_dual_objective
from selection.bayesian.selection_probability_rr import cube_subproblem, cube_gradient, cube_barrier, \
    selection_probability_objective
from selection.bayesian.selection_probability import selection_probability_methods
from selection.randomized.api import randomization
from selection.bayesian.credible_intervals import projected_langevin

class bayesian_inference():
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
                 epsilon):
        (self.y, self.X, self.primal_feasible, self.dual_feasible, self.active, self.active_signs, self.lagrange,
         self.generative_X, self.noise_variance, self.prior_variance, self.randomizer, self.epsilon) =\
            (y, X, primal_feasible, dual_feasible,active, active_signs, lagrange,generative_X, noise_variance,
             prior_variance,randomizer, epsilon)

        self.E = active.sum()
        self.n, self.p = X.shape

    def log_prior(self, true_param):
        return -np.true_divide(true_param.dot(true_param),self.prior_variance)

    def generative_mean(self, true_param):
        return self.generative_X.dot(true_param)

    def likelihood(self, true_param):
        return np.true_divide(np.linalg.norm(self.y - np.squeeze(self.generative_mean(true_param))),
                                        2*self.noise_variance)

    def gradient_selection_prob(self, true_param):

        mean_parameter = np.squeeze(self.generative_mean(true_param))
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
            optimal_dual = mean_parameter - (dual_sol.X_permute.dot(np.linalg.inv(dual_sol.B_p.T))).dot(sel_prob_dual[1])
            return optimal_dual - np.true_divide(mean_parameter, self.noise_variance), sel_prob_dual[0]

        else:
            sel_prob_primal = primal_sol.minimize(max_its=1000, min_its=500, tol=1.e-12)[::-1]
            optimal_primal = (sel_prob_primal[1])[:self.n]
            return optimal_primal- np.true_divide(mean_parameter, self.noise_variance), -sel_prob_primal[0]

    def map_objective(self, true_param):
        return -self.log_prior(true_param) + self.likelihood(true_param) \
               + (self.gradient_selection_prob(true_param)[::-1])[0]

    def selective_map(self):
        initial = np.zeros(self.generative_X.shape[1])
        res = minimize(self.map_objective, x0=initial)
        return res.x

    def posterior_samples(self):
        initial_condition = np.zeros(self.generative_X.shape[1])
        gradient_map = lambda x : self.gradient_selection_prob(x)[0]
        projection_map = lambda x : x
        stepsize = 1./np.sqrt(self.p)
        samples = projected_langevin(initial_condition, gradient_map, projection_map, stepsize)
        return samples


















