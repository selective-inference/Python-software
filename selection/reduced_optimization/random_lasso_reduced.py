import numpy as np
import sys

import regreg.api as rr
from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability
from selection.bayesian.credible_intervals import projected_langevin

class log_likelihood(rr.smooth_atom):

    def __init__(self,
                 mean,
                 Sigma,
                 m,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        initial = np.zeros(m)

        self.mean = mean

        self.Sigma = Sigma

        rr.smooth_atom.__init__(self,
                                (m,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        f = ((arg-self.mean).T.dot(np.linalg.inv(self.Sigma)).dot(arg-self.mean))/2.

        g = (np.linalg.inv(self.Sigma)).dot(arg-self.mean)

        if mode == 'func':
            return f

        elif mode == 'grad':
            return g

        elif mode == 'both':
            return f, g

        else:
            raise ValueError('mode incorrectly specified')

class selection_probability_random_lasso(rr.smooth_atom):

    def __init__(self,
                 map,
                 generative_mean,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.map = map
        self.q = map.p - map.nactive
        self.r = map.p + map.nactive
        self.p = map.p

        self.inactive_conjugate = self.active_conjugate = map.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = self.map.inactive_lagrange

        rr.smooth_atom.__init__(self,
                                (self.r,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.map.feasible_point,
                                coef=coef)

        self.coefs[:] = self.map.feasible_point

        nonnegative = nonnegative_softmax_scaled(self.map.nactive)

        opt_vars = np.zeros(self.r, bool)
        opt_vars[map.p:] = 1

        self._opt_selector = rr.selector(opt_vars, (self.r,))
        self._response_selector = rr.selector(~opt_vars, (self.r,))

        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)

        self.active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                                 rr.affine_transform(np.hstack([self.map.A_active, self.map.B_active]),
                                                                     self.map.offset_active))

        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale=1.)
        self.cube_loss = rr.affine_smooth(cube_obj, np.hstack([self.map.A_inactive, self.map.B_inactive]))

        # w_1, v_1 = np.linalg.eig(self.map.score_cov)
        # self.score_cov_inv_half = (v_1.T.dot(np.diag(np.power(w_1, -0.5)))).dot(v_1)
        # likelihood_loss = rr.signal_approximator(np.squeeze(np.zeros(self.p)), coef=1.)
        # scaled_response_selector = rr.selector(~opt_vars, (self.r,), rr.affine_transform(self.score_cov_inv_half,
        #                                                                                  self.score_cov_inv_half.
        #                                                                                  dot(np.squeeze(generative_mean))))
        #print("cov", self.map.score_cov.shape )
        likelihood_loss = log_likelihood(generative_mean, self.map.score_cov, self.p)

        self.likelihood_loss = rr.affine_smooth(likelihood_loss, self._response_selector)

        self.total_loss = rr.smooth_sum([self.active_conj_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier,
                                         self.cube_loss])

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
            f, g = self.total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, step=1, nstep=100, tol=1.e-8):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current)
            #print("gradient", newton_step)

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                #print("proposal", proposal[self.p:])
                #print("T/F", np.all(proposal[self.p:] > 0))
                if np.all(proposal[self.p:] > 0):
                    #print("here")
                    break
                step *= 0.5
                if count >= 40:
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print("here check")
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


        value = objective(current)

        return current, value


class sel_inf_random_lasso(rr.smooth_atom):

    def __init__(self, solver, prior_variance, coef=1., offset=None, quadratic=None):

        self.solver = solver

        X, _ = self.solver.loss.data
        self.p_shape = X.shape[1]
        self.param_shape = self.solver._overall.sum()
        self.prior_variance = prior_variance

        initial = self.solver.initial_soln[self.solver._overall]
        print("initial_state", initial)

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.initial_state = initial

    def smooth_objective_post(self, sel_param, mode='both', check_feasibility=False):

        sel_param = self.apply_offset(sel_param)
        generative_mean = np.zeros(self.p_shape)
        generative_mean[:self.param_shape] = sel_param

        cov_data_inv = self.solver.score_cov_inv

        sel_lasso = selection_probability_random_lasso(self.solver, generative_mean)

        sel_prob_primal = sel_lasso.minimize2(nstep=100)[::-1]

        optimal_primal = (sel_prob_primal[1])[:self.p_shape]

        sel_prob_val = -sel_prob_primal[0]

        full_gradient = cov_data_inv.dot(optimal_primal - generative_mean)

        optimizer = full_gradient[:self.param_shape]

        likelihood_loss = log_likelihood(self.solver.observed_score_state, self.solver.score_cov, self.p_shape)

        likelihood_loss = rr.affine_smooth(likelihood_loss, np.vstack([np.identity(self.param_shape),
                                                                       np.zeros((self.p_shape -self.param_shape, self.param_shape))]))

        likelihood_loss_value = likelihood_loss.smooth_objective(sel_param, 'func')

        likelihood_loss_grad = likelihood_loss.smooth_objective(sel_param, 'grad')

        log_prior_loss = rr.signal_approximator(np.zeros(self.param_shape), coef=1. / self.prior_variance)

        log_prior_loss_value = log_prior_loss.smooth_objective(sel_param, 'func')

        log_prior_loss_grad = log_prior_loss.smooth_objective(sel_param, 'grad')

        f = likelihood_loss_value + log_prior_loss_value + sel_prob_val

        g = likelihood_loss_grad + log_prior_loss_grad + optimizer

        if mode == 'func':
            return self.scale(f)
        elif mode == 'grad':
            return self.scale(g)
        elif mode == 'both':
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def map_solve(self, step=1, nstep=100, tol=1.e-5):

        current = self.coefs[:]
        current_value = np.inf

        objective = lambda u: self.smooth_objective_post(u, 'func')
        grad = lambda u: self.smooth_objective_post(u, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current)

            # make sure proposal is a descent
            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                # print("proposal", proposal)

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

    def posterior_samples(self, Langevin_steps=1500, burnin=100):
        state = self.initial_state
        print("here", state.shape)
        gradient_map = lambda x: -self.smooth_objective_post(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.param_shape
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in range(Langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            #print i, sampler.state.copy()

        samples = np.array(samples)
        return samples[burnin:, :]

