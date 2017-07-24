import numpy as np
import sys

import regreg.api as rr

from .barrier import barrier_conjugate_softmax_scaled_rr
from .credible_intervals import projected_langevin

class selection_probability_lasso_dual(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,
                 active,  # the active set chosen by randomized lasso
                 active_sign,  # the set of signs of active coordinates chosen by lasso
                 lagrange,  # in R^p
                 mean_parameter,  # in R^n
                 noise_variance,  # noise_level in data
                 randomizer,  # specified randomization
                 epsilon,  # ridge penalty for randomized lasso
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        n, p = X.shape
        E = active.sum()
        self._X = X
        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.CGF_randomization = randomizer.CGF

        if self.CGF_randomization is None:
            raise ValueError(
                'randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        initial = feasible_point

        self.feasible_point = feasible_point

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = feasible_point

        mean_parameter = np.squeeze(mean_parameter)

        self.active = active

        X_E = self.X_E = X[:, active]
        self.X_permute = np.hstack([self.X_E, self._X[:, ~active]])
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.active_slice = np.zeros_like(active, np.bool)
        self.active_slice[:active.sum()] = True

        self.B_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_sign[None, :], np.zeros((E, p - E))])
        self.B_inactive = np.hstack([B_mE * active_sign[None, :], np.identity((p - E))])
        self.B_p = np.vstack((self.B_active, self.B_inactive))

        self.B_p_inv = np.linalg.inv(self.B_p.T)

        self.offset_active = active_sign * lagrange[active]
        self.inactive_subgrad = np.zeros(p - E)

        self.cube_bool = np.zeros(p, np.bool)

        self.cube_bool[E:] = 1

        self.dual_arg = self.B_p_inv.dot(np.append(self.offset_active, self.inactive_subgrad))

        self._opt_selector = rr.selector(~self.cube_bool, (p,))

        self.set_parameter(mean_parameter, noise_variance)

        _barrier_star = barrier_conjugate_softmax_scaled_rr(self.cube_bool, self.inactive_lagrange)

        self.conjugate_barrier = rr.affine_smooth(_barrier_star, np.identity(p))

        self.CGF_randomizer = rr.affine_smooth(self.CGF_randomization, -self.B_p_inv)

        self.constant = np.true_divide(mean_parameter.dot(mean_parameter), 2 * noise_variance)

        self.linear_term = rr.identity_quadratic(0, 0, self.dual_arg, -self.constant)

        self.total_loss = rr.smooth_sum([self.conjugate_barrier,
                                         self.CGF_randomizer,
                                         self.likelihood_loss])

        self.total_loss.quadratic = self.linear_term

    def set_parameter(self, mean_parameter, noise_variance):

        mean_parameter = np.squeeze(mean_parameter)

        self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)

        self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, self.X_permute.dot(self.B_p_inv))

    def _smooth_objective(self, param, mode='both', check_feasibility=False):

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

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        n, p = self._X.shape

        current = self.feasible_point
        current_value = np.inf

        objective = lambda u: self.total_loss.objective(u)
        grad = lambda u: self.total_loss.smooth_objective(u, 'grad') + self.dual_arg

        for itercount in xrange(nstep):
            newton_step = grad(current) * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.all(proposal[self.active_slice] < 0):
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


class sel_prob_gradient_map_lasso(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,  # in R^{ |E|}
                 active,
                 active_sign,
                 lagrange,  # in R^p
                 generative_X,  # in R^{p}\times R^{n}
                 noise_variance,
                 randomizer,
                 epsilon,  # ridge penalty for randomized lasso
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.E = active.sum()
        self.n, self.p = X.shape
        self.dim = generative_X.shape[1]

        self.noise_variance = noise_variance

        (self.X, self.feasible_point, self.active, self.active_sign, self.lagrange, self.generative_X, self.noise_variance,
         self.randomizer, self.epsilon) = (X, feasible_point, active, active_sign, lagrange, generative_X,
                                           noise_variance, randomizer, epsilon)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-8):

        true_param = self.apply_offset(true_param)

        mean_parameter = np.squeeze(self.generative_X.dot(true_param))

        dual_sol = selection_probability_lasso_dual(self.X,
                                                    self.feasible_point,
                                                    self.active,
                                                    self.active_sign,
                                                    self.lagrange,
                                                    mean_parameter,
                                                    self.noise_variance,
                                                    self.randomizer,
                                                    self.epsilon)

        sel_prob_dual = dual_sol.minimize2(nstep=100)[::-1]
        optimal_dual = mean_parameter - (dual_sol.X_permute.dot(np.linalg.inv(dual_sol.B_p.T))).dot(sel_prob_dual[1])
        sel_prob_val = sel_prob_dual[0]
        optimizer = self.generative_X.T.dot(np.true_divide(optimal_dual - mean_parameter, self.noise_variance))

        if mode == 'func':
            return sel_prob_val
        elif mode == 'grad':
            return optimizer
        elif mode == 'both':
            return sel_prob_val, optimizer
        else:
            raise ValueError('mode incorrectly specified')


class selective_inf_lasso(rr.smooth_atom):
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

        self.E = grad_map.E

        self.generative_X = grad_map.generative_X

        initial = np.zeros(self.E)

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

        self.total_loss_0 = rr.smooth_sum([self.likelihood_loss,
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
            f = self.total_loss_0.smooth_objective(true_param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss_0.smooth_objective(true_param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = self.total_loss_0.smooth_objective(true_param, 'both')
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

    def posterior_samples(self, Langevin_steps=1500, burnin=50):
        state = self.initial_state
        sys.stderr.write("Number of selected variables by randomized lasso: "+str(state.shape)+"\n")
        gradient_map = lambda x: -self.smooth_objective(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.E
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in xrange(Langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            #print i, sampler.state.copy()
            sys.stderr.write("sample number: " + str(i)+"\n")

        samples = np.array(samples)
        return samples[burnin:, :]

    def posterior_risk(self, estimator_1, estimator_2, Langevin_steps=2000, burnin=0):
        state = self.initial_state
        sys.stderr.write("Number of selected variables by randomized lasso: "+str(state.shape)+"\n")
        gradient_map = lambda x: -self.smooth_objective(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.E
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        post_risk_1 = 0.
        post_risk_2 = 0.

        for i in range(Langevin_steps):
            sampler.next()
            sample = sampler.state.copy()

            #print(sample)
            risk_1 = ((estimator_1-sample)**2).sum()
            print("adjusted risk", risk_1)
            post_risk_1 += risk_1

            risk_2 = ((estimator_2-sample) ** 2).sum()
            print("unadjusted risk", risk_2)
            post_risk_2 += risk_2


        return post_risk_1/Langevin_steps, post_risk_2/Langevin_steps




