import numpy as np
import sys
import regreg.api as rr
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.cisEQTLS.lasso_reduced import neg_log_cube_probability
from selection.bayesian.credible_intervals import projected_langevin


class selection_probability_genes_variants(rr.smooth_atom):

    def __init__(self,
                 X, #matrix of SNPs per gene in a sample of n individuals
                 feasible_point,  # in R^{1 + |E|}, |E| size of set chosen by lasso
                 index,  # the smallest index i_0 such that the corresponding ordered (t_0) p-value is smaller than (t_0+1)alpha/2p
                 J, #the set of all indices corresponding to ordered statistics smaller than (t_0)
                 active,  # the active set chosen by randomized lasso
                 T_sign,  # the sign of 2-sided T statistic corresponding to index active_1
                 active_sign,  # the set of signs of active coordinates chosen by lasso
                 lagrange,  # in R^p
                 threshold,  # in R^{t_0+1}
                 mean_parameter,  # in R^n
                 noise_variance, #noise_level in data
                 randomizer,
                 epsilon,  # ridge penalty for randomized lasso
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        n, p = X.shape

        self._X = X

        E = active.sum()
        self.q = p - E

        sigma = np.sqrt(noise_variance)

        self.index = index
        self.J = J
        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer
        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = np.zeros(n + 1 + E, )
        initial[n:] = feasible_point
        self.n = n

        rr.smooth_atom.__init__(self,
                                (n + 1 + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        #print("initial", self.coefs)

        opt_vars = np.zeros(n + 1 + E, bool)
        opt_vars[n:] = 1

        nonnegative = nonnegative_softmax(1 + E)

        self._opt_selector = rr.selector(opt_vars, (n + 1 + E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + 1 + E,))

        self.set_parameter(mean_parameter, noise_variance)

        arg_simes = np.zeros(self.n + 1 + E, bool)
        arg_simes[:self.n + 1] = 1
        arg_lasso = np.zeros(self.n + 1, bool)
        arg_lasso[:self.n] = 1
        arg_lasso = np.append(arg_lasso, np.ones(E, bool))

        #print("shapes", np.true_divide(-X[:, index], sigma)[None,:].shape, (np.identity(1)* T_sign[None, :]).shape, index)

        self.A_active_1 = np.hstack([np.true_divide(-X[:, index], sigma)[None,:], np.identity(1)* T_sign[None, :]])

        #print("scalar", T_sign, threshold[-1])

        self.offset_active_1 = T_sign * threshold[-1]

        self._active_simes = rr.selector(arg_simes, (self.n + 1 + E,),
                                         rr.affine_transform(self.A_active_1, self.offset_active_1))

        self.active_conj_loss_1 = rr.affine_smooth(self.active_conjugate, self._active_simes)

        X_E = X[:, active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active_2 = np.hstack([-X[:, active].T, (B_E + epsilon * np.identity(E)) * active_sign[None, :]])

        self.A_inactive_2 = np.hstack([-X[:, ~active].T, (B_mE * active_sign[None, :])])

        self.offset_active_2 = active_sign * lagrange[active]

        self.offset_inactive_2 = np.zeros(p - E)

        self._active_lasso = rr.selector(arg_lasso, (self.n + 1 + E,),
                                         rr.affine_transform(self.A_active_2, self.offset_active_2))

        self._inactive_lasso = rr.selector(arg_lasso, (self.n + 1 + E,),
                                           rr.affine_transform(self.A_inactive_2, self.offset_inactive_2))

        self.active_conj_loss_2 = rr.affine_smooth(self.active_conjugate, self._active_lasso)

        cube_obj_2 = neg_log_cube_probability(self.q, lagrange[~active], randomization_scale = 1.)

        self.cube_loss_2 = rr.affine_smooth(cube_obj_2, self._inactive_lasso)

        if threshold.shape[0] > 1:

            J_card = J.shape[0]
            self.A_inactive_1 = np.hstack([np.true_divide(-X[:, J].T, sigma), np.zeros((J_card, 1))])
            self.offset_inactive_1 = np.zeros(J_card)
            self._inactive_simes = rr.selector(arg_simes, (self.n + 1 + E,),
                                               rr.affine_transform(self.A_inactive_1, self.offset_inactive_1))

            cube_obj_1 = neg_log_cube_probability(J_card, threshold[:J_card], randomization_scale = 1.)

            self.cube_loss_1 = rr.affine_smooth(cube_obj_1, self._inactive_simes)

            self.total_loss = rr.smooth_sum([self.active_conj_loss_1,
                                             self.active_conj_loss_2,
                                             self.cube_loss_1,
                                             self.cube_loss_2,
                                             self.likelihood_loss,
                                             self.nonnegative_barrier])

        else:

            self.total_loss = rr.smooth_sum([self.active_conj_loss_1,
                                             self.active_conj_loss_2,
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

    def minimize2(self, step=1, nstep=100, tol=1.e-8):

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
                #print("proposal", proposal[n:])
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


class sel_prob_gradient_map_simes_lasso(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,  # in R^{1 + |E|}
                 index,
                 J,
                 active,
                 T_sign,
                 active_sign,
                 lagrange,  # in R^p
                 threshold,  # in R^p
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

        (self.X, self.feasible_point, self.index, self.J, self.active, self.T_sign, self.active_sign,
         self.lagrange, self.threshold, self.generative_X, self.noise_variance, self.randomizer, self.epsilon) \
            = (X, feasible_point, index, J, active, T_sign, active_sign, lagrange,
               threshold, generative_X, noise_variance, randomizer, epsilon)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):
        true_param = self.apply_offset(true_param)

        mean_parameter = np.squeeze(self.generative_X.dot(true_param))

        primal_sol = selection_probability_genes_variants(self.X,
                                                          self.feasible_point,
                                                          self.index,
                                                          self.J,
                                                          self.active,
                                                          self.T_sign,
                                                          self.active_sign,
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


class selective_inf_simes_lasso(rr.smooth_atom):
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






