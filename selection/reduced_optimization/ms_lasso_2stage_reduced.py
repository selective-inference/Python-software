import numpy as np
import sys
from scipy.stats import norm

import regreg.api as rr

from .credible_intervals import projected_langevin
from .lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability

class selection_probability_objective_ms_lasso(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point, #in R^{|E|_1 + |E|_2}
                 active_1, #the active set chosen by randomized marginal screening
                 active_2, #the active set chosen by randomized lasso
                 active_signs_1, #the set of signs of active coordinates chosen by ms
                 active_signs_2, #the set of signs of active coordinates chosen by lasso
                 lagrange, #in R^p
                 threshold, #in R^p
                 mean_parameter, # in R^n
                 noise_variance,
                 randomizer,
                 epsilon, #ridge penalty for randomized lasso
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        n, p = X.shape
        self._X = X

        E_1 = active_1.sum()
        E_2 = active_2.sum()

        sigma = np.sqrt(noise_variance)

        self.active_1 = active_1
        self.active_2 = active_2
        self.noise_variance = noise_variance
        self.randomization = randomizer
        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = np.zeros(n + E_1 + E_2, )
        initial[n:] = feasible_point
        self.n = n

        rr.smooth_atom.__init__(self,
                                (n + E_1 + E_2,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial
        nonnegative = nonnegative_softmax_scaled(E_1 + E_2)
        opt_vars = np.zeros(n + E_1 + E_2, bool)
        opt_vars[n:] = 1

        self._opt_selector = rr.selector(opt_vars, (n + E_1 + E_2,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + E_1 + E_2,))

        self.set_parameter(mean_parameter, noise_variance)

        arg_ms = np.zeros(self.n + E_1 + E_2, bool)
        arg_ms[:self.n + E_1] = 1
        arg_lasso = np.zeros(self.n + E_1, bool)
        arg_lasso[:self.n] = 1
        arg_lasso = np.append(arg_lasso, np.ones(E_2, bool))

        self.A_active_1 = np.hstack([np.true_divide(-X[:, active_1].T, sigma), np.identity(E_1)
                                     * active_signs_1[None, :]])

        self.A_inactive_1 = np.hstack([np.true_divide(-X[:, ~active_1].T, sigma), np.zeros((p - E_1, E_1))])

        self.offset_active_1 = active_signs_1 * threshold[active_1]
        self.offset_inactive_1 = np.zeros(p - E_1)

        self._active_ms = rr.selector(arg_ms, (self.n + E_1 + E_2,),
                                      rr.affine_transform(self.A_active_1, self.offset_active_1))

        self._inactive_ms = rr.selector(arg_ms, (self.n + E_1 + E_2,),
                                        rr.affine_transform(self.A_inactive_1, self.offset_inactive_1))

        self.active_conj_loss_1 = rr.affine_smooth(self.active_conjugate, self._active_ms)

        self.q_1 = p - E_1

        cube_obj_1 = neg_log_cube_probability(self.q_1, threshold[~active_1], randomization_scale=1.)

        self.cube_loss_1 = rr.affine_smooth(cube_obj_1, self._inactive_ms)

        X_step2 = X[:, active_1]
        X_E_2 = X_step2[:, active_2]
        B = X_step2.T.dot(X_E_2)

        B_E = B[active_2]
        B_mE = B[~active_2]

        self.A_active_2 = np.hstack(
            [-X_step2[:, active_2].T, (B_E + epsilon * np.identity(E_2)) * active_signs_2[None, :]])
        self.A_inactive_2 = np.hstack([-X_step2[:, ~active_2].T, (B_mE * active_signs_2[None, :])])

        self.offset_active_2 = active_signs_2 * lagrange[active_2]

        self.offset_inactive_2 = np.zeros(E_1 - E_2)

        self._active_lasso = rr.selector(arg_lasso, (self.n + E_1 + E_2,),
                                         rr.affine_transform(self.A_active_2, self.offset_active_2))

        self._inactive_lasso = rr.selector(arg_lasso, (self.n + E_1 + E_2,),
                                           rr.affine_transform(self.A_inactive_2, self.offset_inactive_2))

        self.active_conj_loss_2 = rr.affine_smooth(self.active_conjugate, self._active_lasso)

        self.q_2 = E_1 - E_2

        cube_obj_2 = neg_log_cube_probability(self.q_2, lagrange[~active_2], randomization_scale=1.)

        self.cube_loss_2 = rr.affine_smooth(cube_obj_2, self._inactive_lasso)

        self.total_loss = rr.smooth_sum([self.active_conj_loss_1,
                                         self.active_conj_loss_2,
                                         self.cube_loss_1,
                                         self.cube_loss_2,
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
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

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


class sel_prob_gradient_map_ms_lasso(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,  # in R^{|E|_1 + |E|_2}
                 active_1,  # the active set chosen by randomized marginal screening
                 active_2,  # the active set chosen by randomized lasso
                 active_signs_1,  # the set of signs of active coordinates chosen by ms
                 active_signs_2,  # the set of signs of active coordinates chosen by lasso
                 lagrange,  # in R^p
                 threshold,  # in R^p
                 generative_X,  # in R^{p}\times R^{n}
                 noise_variance,
                 randomizer,
                 epsilon,  # ridge penalty for randomized lasso
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.E_1 = active_1.sum()
        self.E_2 = active_2.sum()
        self.n, self.p = X.shape
        self.dim = generative_X.shape[1]

        self.noise_variance = noise_variance

        (self.X, self.feasible_point, self.active_1, self.active_2, self.active_signs_1, self.active_signs_2,
         self.lagrange, self.threshold, self.generative_X, self.noise_variance, self.randomizer, self.epsilon) \
            = (X, feasible_point, active_1, active_2, active_signs_1, active_signs_2, lagrange,
               threshold, generative_X, noise_variance, randomizer, epsilon)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):
        true_param = self.apply_offset(true_param)

        mean_parameter = np.squeeze(self.generative_X.dot(true_param))

        primal_sol = selection_probability_objective_ms_lasso(self.X,
                                                              self.feasible_point,
                                                              self.active_1,
                                                              self.active_2,
                                                              self.active_signs_1,
                                                              self.active_signs_2,
                                                              self.lagrange,
                                                              self.threshold,
                                                              mean_parameter,
                                                              self.noise_variance,
                                                              self.randomizer,
                                                              self.epsilon)

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


class selective_map_credible_ms_lasso(rr.smooth_atom):
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

        E_1 = grad_map.E_1

        E_2 = grad_map.E_2

        self.E = E_2

        self.generative_X = grad_map.generative_X

        initial = np.zeros(self.E)

        #initial[:E_1] = np.squeeze(grad_map.feasible_point[:E_1]* grad_map.active_signs_1[None,:])

        initial = np.squeeze(grad_map.feasible_point[E_1:]* grad_map.active_signs_2[None,:])

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

    def posterior_samples(self, langevin_steps=1500, burnin=50):
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

    def posterior_risk(self, estimator_1, estimator_2, langevin_steps=1200, burnin=0):
        state = self.initial_state
        print("here", state.shape)
        gradient_map = lambda x: -self.smooth_objective(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.E
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        post_risk_1 = 0.
        post_risk_2 = 0.

        for i in range(langevin_steps):
            sampler.next()
            sample = sampler.state.copy()

            #print(sample)
            risk_1 = ((estimator_1-sample)**2).sum()
            print("adjusted risk", risk_1)
            post_risk_1 += risk_1

            risk_2 = ((estimator_2-sample) ** 2).sum()
            print("unadjusted risk", risk_2)
            post_risk_2 += risk_2


        return post_risk_1/langevin_steps, post_risk_2/langevin_steps




