import numpy as np
import regreg.api as rr
from selection.bayesian.randomX_lasso_primal import selection_probability_objective_randomX

class sel_prob_gradient_map(rr.smooth_atom):
    def __init__(self,
                 X,
                 primal_feasible,
                 active,
                 active_signs,
                 lagrange,
                 Sigma_parameter,
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

        self.noise_variance = noise_variance

        (self.X, self.primal_feasible, self.active, self.active_signs, self.lagrange, self.Sigma_parameter,
         self.generative_X, self.noise_variance, self.randomizer, self.epsilon) = (X, primal_feasible, active,
                                                                                   active_signs, lagrange,
                                                                                   Sigma_parameter,
                                                                                   generative_X, noise_variance,
                                                                                   randomizer, epsilon)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

        X_active = X[:, active]
        self.X_inactive = X[:, ~active]
        self.X_gen_inv = np.linalg.pinv(X_active)
        self.X_projection = X_active.dot(self.X_gen_inv)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):

        true_param = self.apply_offset(true_param)

        generative_mean = np.squeeze(self.generative_X.dot(true_param))

        mean_suff = self.X_gen_inv.dot(generative_mean)
        mean_nuisance = ((self.X_inactive.T).dot((np.identity(self.n) - self.X_projection))).dot(generative_mean)
        mean_parameter = np.append(mean_suff, mean_nuisance)

        primal_sol = selection_probability_objective_randomX(self.X,
                                                             self.primal_feasible,
                                                             self.active,
                                                             self.active_signs,
                                                             self.lagrange,
                                                             mean_parameter,
                                                             self.Sigma_parameter,
                                                             self.noise_variance,
                                                             self.randomizer,
                                                             self.epsilon)


        sel_prob_primal = primal_sol.minimize2(nstep=100)[::-1]
        optimal_primal = (sel_prob_primal[1])[:self.p]
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