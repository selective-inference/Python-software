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

        self.noise_variance = noise_variance

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

        #if self.n + self.E > self.p:
            #sel_prob_dual = dual_sol.minimize2(nstep=60)[::-1]
            #optimal_dual = mean_parameter - (dual_sol.X_permute.dot(np.linalg.inv(dual_sol.B_p.T))).\
            #    dot(sel_prob_dual[1])
            #sel_prob_val = sel_prob_dual[0]
            #optimizer = self.generative_X.T.dot(np.true_divide(optimal_dual - mean_parameter, self.noise_variance))

        #else:
        sel_prob_primal = primal_sol.minimize2(nstep=60)[::-1]
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

class selective_map_credible(rr.smooth_atom):
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

        self.E = active.sum()

        self.generative_X = generative_X

        if self.param_shape == self.E:
            initial = np.squeeze(primal_feasible * active_signs[None,:])
        else:
            initial = np.zeros(self.param_shape)

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.initial_state = initial

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

        self.noise_variance = noise_variance

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

    def map_solve_2(self, step=1, nstep=500, tol=1.e-8):

        current = self.coefs[:]
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current)* self.noise_variance

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

        value = objective(current)
        return current, value

    def map_solve(self, initial=None, min_its=10, max_its=50, tol=1.e-10):

        problem = rr.separable_problem.singleton(self)
        #problem.coefs[:] = 0.5
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        value = problem.objective(soln)
        return soln, value

    def posterior_samples(self, Langevin_steps = 3000, burnin = 100):
        state = self.initial_state
        gradient_map = lambda x: -self.smooth_objective(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.E
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in range(Langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            print i, sampler.state.copy()

        samples = np.array(samples)
        return samples[burnin:, :]















