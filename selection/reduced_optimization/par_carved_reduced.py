import numpy as np
import sys

import regreg.api as rr
from .lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability
from .credible_intervals import projected_langevin
from .par_random_lasso_reduced import log_likelihood

class smooth_cube_barrier(rr.smooth_atom):

    def __init__(self,
                 lagrange_cube,  # cube half lengths
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.lagrange_cube = lagrange_cube

        rr.smooth_atom.__init__(self,
                                (self.lagrange_cube.shape[0],),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)
        #BIG = 10 ** 10
        _diff = arg - self.lagrange_cube  # z - \lambda < 0
        _sum = arg + self.lagrange_cube  # z + \lambda > 0
        #violations = ((_diff >= 0).sum() + (_sum <= 0).sum() > 0)

        f = np.log((_diff - 1.) * (_sum + 1.) / (_diff * _sum)).sum() #+ BIG * violations
        g = 1. / (_diff - 1) - 1. / _diff + 1. / (_sum + 1) - 1. / _sum

        if mode == 'func':
            return self.scale(f)
        elif mode == 'grad':
            return self.scale(g)
        elif mode == 'both':
            return self.scale(f), self.scale(g)
        else:
            raise ValueError('mode incorrectly specified')


class selection_probability_carved(rr.smooth_atom):

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

        rr.smooth_atom.__init__(self,
                                (2*self.p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.map.feasible_point,
                                coef=coef)

        self.coefs[:] = self.map.feasible_point


        opt_vars_0 = np.zeros(self.r, bool)
        opt_vars_0[self.p:] = 1
        opt_vars = np.append(opt_vars_0, np.ones(self.q, bool))

        opt_vars_active = np.append(opt_vars_0, np.zeros(self.q, bool))
        opt_vars_inactive = np.zeros(2 * self.p, bool)
        opt_vars_inactive[self.r:] = 1

        self._response_selector = rr.selector(~opt_vars, (2 * self.p,))
        self._opt_selector_active = rr.selector(opt_vars_active, (2 * self.p,))
        self._opt_selector_inactive = rr.selector(opt_vars_inactive, (2 * self.p,))

        nonnegative = nonnegative_softmax_scaled(self.map.nactive)
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector_active)

        cube_objective = smooth_cube_barrier(self.map.inactive_lagrange)
        self.cube_barrier = rr.affine_smooth(cube_objective, self._opt_selector_inactive)


        linear_map = np.hstack([self.map._score_linear_term, self.map._opt_linear_term])
        randomization_loss = log_likelihood(np.zeros(self.p), self.map.randomization_cov, self.p)
        self.randomization_loss = rr.affine_smooth(randomization_loss, rr.affine_transform(linear_map,
                                                                                           self.map._opt_affine_term))

        likelihood_loss = log_likelihood(generative_mean, self.map.score_cov, self.p)

        self.likelihood_loss = rr.affine_smooth(likelihood_loss, self._response_selector)

        self.total_loss = rr.smooth_sum([self.randomization_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier,
                                         self.cube_barrier])

    def smooth_objective(self, param, mode='both', check_feasibility=False):

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

        for itercount in xrange(nstep):
            newton_step = grad(current)
            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                proposal_opt = proposal[self.p:]
                failing_cube = (proposal[self.r:] > self.map.inactive_lagrange) + \
                               (proposal[self.r:] < - self.map.inactive_lagrange)

                failing_sign = (proposal_opt[:self.map.nactive] < 0)
                failing = failing_cube.sum() + failing_sign.sum()

                if not failing:
                    break
                step *= 0.5

                if count >= 60:
                    raise ValueError('not finding a feasible point')

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


class sel_inf_carved(rr.smooth_atom):

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

        sel_lasso = selection_probability_carved(self.solver, generative_mean)

        sel_prob_primal = sel_lasso.minimize2(nstep=100)[::-1]

        optimal_primal = (sel_prob_primal[1])[:self.p_shape]

        sel_prob_val = -sel_prob_primal[0]

        full_gradient = cov_data_inv.dot(optimal_primal - generative_mean)

        optimizer = full_gradient[:self.param_shape]

        likelihood_loss = log_likelihood(self.solver.target_observed, self.solver.score_cov[:self.param_shape,
                                                                      :self.param_shape], self.param_shape)

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

    def posterior_samples(self, langevin_steps=1500, burnin=100):
        state = self.initial_state
        print("here", state.shape)
        gradient_map = lambda x: -self.smooth_objective_post(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.param_shape
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in xrange(langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            sys.stderr.write("sample number: " + str(i) + "\n")

        samples = np.array(samples)
        return samples[burnin:, :]






