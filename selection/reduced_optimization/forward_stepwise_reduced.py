from math import log
import sys
import numpy as np
import regreg.api as rr
from scipy.stats import norm

from .lasso_reduced import nonnegative_softmax_scaled
from .credible_intervals import projected_langevin

class neg_log_cube_probability_fs(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 p,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.q = q
        self.p = p

        rr.smooth_atom.__init__(self,
                                (self.q+1,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, par, mode='both', check_feasibility=False, tol=1.e-6):

        par = self.apply_offset(par)

        mu = par[:self.p-1]
        arg = par[self.p-1]

        arg_u = ((arg *np.ones(self.q)) + mu) / self.randomization_scale
        arg_l = (-(arg *np.ones(self.q)) + mu) / self.randomization_scale
        prod_arg = np.exp(-(2. * mu * (arg *np.ones(self.q))) / (self.randomization_scale ** 2))
        neg_prod_arg = np.exp((2. * mu * (arg *np.ones(self.q))) / (self.randomization_scale ** 2))

        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()

        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(mu > 0)] = 1
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


class selection_probability_objective_fs(rr.smooth_atom):

    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_sign,
                 mean_parameter,  # in R^n
                 noise_variance,
                 randomizer,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):


        self.n, p = X.shape
        E = 1
        self.q = p-1
        self._X = X
        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = np.zeros(self.n + E, )
        initial[self.n:] = feasible_point

        rr.smooth_atom.__init__(self,
                                (self.n + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        nonnegative = nonnegative_softmax_scaled(E)

        opt_vars = np.zeros(self.n + E, bool)
        opt_vars[self.n:] = 1

        self._opt_selector = rr.selector(opt_vars, (self.n + E,))
        self._response_selector = rr.selector(~opt_vars, (self.n + E,))

        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)

        sign = np.zeros((1, 1))
        sign[0:, :] = active_sign
        self.A_active = np.hstack([-X[:, active].T, sign])
        self.active_conj_loss = rr.affine_smooth(self.active_conjugate, self.A_active)

        self.A_in_1 = np.hstack([-X[:, ~active].T, np.zeros((p - 1, 1))])
        self.A_in_2 = np.hstack([np.zeros((self.n, 1)).T, np.ones((1, 1))])
        self.A_inactive = np.vstack([self.A_in_1, self.A_in_2])

        cube_loss = neg_log_cube_probability_fs(self.q, p)
        self.cube_loss = rr.affine_smooth(cube_loss, self.A_inactive)

        self.set_parameter(mean_parameter, noise_variance)

        self.total_loss = rr.smooth_sum([self.active_conj_loss,
                                         self.cube_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier])

    def set_parameter(self, mean_parameter, noise_variance):
        """
        Set $\beta_E^*$.
        """
        mean_parameter = np.squeeze(mean_parameter)
        likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)
        self.likelihood_loss = rr.affine_smooth(likelihood_loss, self._response_selector)

    def smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        if mode == 'func':
            f = self.total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f = self.total_loss.smooth_objective(param, 'func')
            g = self.total_loss.smooth_objective(param, 'grad')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        n, p = self._X.shape

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current) * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.all(proposal[n:] > 0):
                    break
                step *= 0.5
                if count >= 40:
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                # print(current_value, proposed_value, 'minimize')
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

        # print('iter', itercount)
        value = objective(current)
        return current, value


class sel_prob_gradient_map_fs(rr.smooth_atom):
    def __init__(self,
                 X,
                 primal_feasible,
                 active,
                 active_sign,
                 generative_X,
                 noise_variance,
                 randomizer,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.E = 1
        self.n, self.p = X.shape
        self.dim = generative_X.shape[1]

        self.noise_variance = noise_variance

        (self.X, self.primal_feasible, self.active, self.active_sign, self.generative_X, self.noise_variance,
         self.randomizer) = (X, primal_feasible, active, active_sign, generative_X, noise_variance, randomizer)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):

        true_param = self.apply_offset(true_param)

        mean_parameter = np.squeeze(self.generative_X.dot(true_param))

        primal_sol = selection_probability_objective_fs(self.X,
                                                        self.primal_feasible,
                                                        self.active,
                                                        self.active_sign,
                                                        mean_parameter,
                                                        self.noise_variance,
                                                        self.randomizer)

        sel_prob_primal = primal_sol.minimize2(nstep=100)[::-1]
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

class selective_map_credible_fs(rr.smooth_atom):
    def __init__(self,
                 y,
                 grad_map,
                 prior_variance,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        generative_X = grad_map.generative_X
        self.param_shape = generative_X.shape[1]

        y = np.squeeze(y)

        self.E = 1

        self.generative_X = grad_map.generative_X

        initial = np.zeros(1)

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        noise_variance = grad_map.noise_variance

        self.set_likelihood(y, noise_variance, generative_X)

        self.set_prior(prior_variance)

        self.initial_state = initial

        self.total_loss = rr.smooth_sum([self.likelihood_loss,
                                         self.log_prior_loss,
                                         grad_map])

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

    def map_solve(self, step=1, nstep=100, tol=1.e-8):

        current = self.coefs[:]
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current)
            # * self.noise_variance

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

    def posterior_samples(self, langevin_steps=1000, burnin=100):
        state = self.initial_state
        print("here", state.shape)
        gradient_map = lambda x: -self.smooth_objective(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.E
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in range(langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            #print i, sampler.state.copy()
            sys.stderr.write("sample number: " + str(i) + "\n")

        samples = np.array(samples)
        return samples[burnin:, :]
