import numpy as np
import regreg.api as rr
from selection.bayesian.randomX_lasso_primal import selection_probability_objective_randomX
from selection.bayesian.credible_intervals import projected_langevin

class sel_prob_gradient_map_randomX(rr.smooth_atom):
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
        X_inactive = X[:, ~active]
        X_gen_inv = np.linalg.pinv(X_active)
        X_projection = X_active.dot(X_gen_inv)
        X_inter = (X_inactive.T).dot((np.identity(self.n) - X_projection))
        self.D_mean = np.vstack([X_gen_inv,X_inter])

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):

        true_param = self.apply_offset(true_param)

        generative_mean = np.squeeze(self.generative_X.dot(true_param))
        mean_parameter = self.D_mean.dot(generative_mean)

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
        coef = (np.linalg.inv(self.Sigma_parameter)).dot(self.D_mean.dot(self.generative_X))
        optimizer = (coef.T).dot(np.true_divide(optimal_primal - mean_parameter, self.noise_variance))

        if mode == 'func':
            return sel_prob_val
        elif mode == 'grad':
            return optimizer
        elif mode == 'both':
            return sel_prob_val, optimizer
        else:
            raise ValueError('mode incorrectly specified')


class selective_map_credible_randomX(rr.smooth_atom):
    def __init__(self,
                 y,
                 X,
                 primal_feasible,
                 active,
                 active_signs,
                 lagrange,
                 Sigma_parameter,
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
        n = generative_X.shape[0]

        y = np.squeeze(y)

        self.E = active.sum()

        self.generative_X = generative_X

        initial = np.squeeze(primal_feasible * active_signs[None,:])

        X_active = X[:, active]
        X_inactive = X[:, ~active]
        X_gen_inv = np.linalg.pinv(X_active)
        X_projection = X_active.dot(X_gen_inv)
        X_inter = (X_inactive.T).dot((np.identity(n) - X_projection))
        self.D_mean = np.vstack([X_gen_inv, X_inter])

        w, v = np.linalg.eig(Sigma_parameter)
        self.Sigma_inv_half = (v.T.dot(np.diag(np.power(w, -0.5)))).dot(v)

        data_obs = self.D_mean.dot(y)

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.initial_state = np.squeeze(primal_feasible * active_signs[None, :]) #initial_state valid only for selected model

        self.set_likelihood(data_obs, noise_variance, generative_X)

        self.set_prior(prior_variance)

        self.log_sel_prob = sel_prob_gradient_map_randomX(X,
                                                          primal_feasible,
                                                          active,
                                                          active_signs,
                                                          lagrange,
                                                          Sigma_parameter,
                                                          generative_X,
                                                          noise_variance,
                                                          randomizer,
                                                          epsilon)

        self.total_loss = rr.smooth_sum([self.likelihood_loss,
                                         self.log_prior_loss,
                                         self.log_sel_prob])

        self.noise_variance = noise_variance

    def set_likelihood(self, data_obs, noise_variance, generative_X):
        scaled_data_obs = self.Sigma_inv_half.dot(data_obs)
        likelihood_loss = rr.signal_approximator(scaled_data_obs, coef=1. / noise_variance)
        coef_param = self.Sigma_inv_half.dot(self.D_mean.dot(self.generative_X))
        self.likelihood_loss = rr.affine_smooth(likelihood_loss, coef_param)

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

    def map_solve_2(self, step=1, nstep=300, tol=1.e-8):

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

    def posterior_samples(self, Langevin_steps = 5000, burnin = 500):
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