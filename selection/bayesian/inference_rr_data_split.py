
import numpy as np
import regreg.api as rr
from selection.algorithms.softmax import nonnegative_softmax
from selection.randomized.M_estimator import M_estimator_split
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov
from selection.bayesian.credible_intervals import projected_langevin


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

class selection_probability_split(rr.smooth_atom):

    def __init__(self, solver, generative_mean, coef=1., offset=None, quadratic=None):

        self.loss = solver.loss
        self._beta_full = solver._beta_full
        self._overall = solver._overall

        nactive = self._overall.sum()

        X, _ = solver.loss.data
        n, p = X.shape

        lagrange = []
        for key, value in solver.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        self.feasible_point = solver.observed_opt_state

        initial = np.zeros(2*p, )
        initial[p:] = self.feasible_point

        rr.smooth_atom.__init__(self,
                                (2*p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        bootstrap_score, randomization_cov = solver.setup_sampler()

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        score_linear_term = solver.score_transform[0]

        (opt_linear_term, opt_affine_term) = solver.opt_transform

        B = opt_linear_term
        A = score_linear_term

        self.linear_map = np.hstack([A,B])
        gamma = opt_affine_term

        opt_vars_0 = np.zeros(p + nactive, bool)
        opt_vars_0[p:] = 1
        opt_vars = np.append(opt_vars_0, np.ones(p-nactive, bool))
        opt_vars_active = np.append(opt_vars_0, np.zeros(p-nactive, bool))
        opt_vars_inactive = np.zeros(2*p, bool)
        opt_vars_inactive[p+ nactive:] = 1

        self._response_selector = rr.selector(~opt_vars, (2*p,))
        self._opt_selector_active = rr.selector(opt_vars_active, (2*p,))
        self._opt_selector_inactive = rr.selector(opt_vars_inactive, (2*p,))

        nonnegative = nonnegative_softmax(nactive)
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector_active)

        cube_objective = smooth_cube_barrier(self.inactive_lagrange)
        self.cube_barrier = rr.affine_smooth(cube_objective, self._opt_selector_inactive)

        w, v = np.linalg.eig(randomization_cov)
        self.randomization_cov_inv_half = (v.T.dot(np.diag(np.power(w, -0.5)))).dot(v)
        self.randomization_quad = self.randomization_cov_inv_half.dot(self.linear_map)
        self.offset_quad = self.randomization_cov_inv_half.dot(gamma)
        gaussian_loss = rr.signal_approximator(np.zeros(p), coef=1.)
        self.randomization_loss = rr.affine_smooth(gaussian_loss, rr.affine_transform(self.randomization_quad,
                                                                                      self.offset_quad))
        #print("here", self.randomization_quad.shape, self.offset_quad.shape)

        w_1, v_1 = np.linalg.eig(score_cov)
        self.score_cov_inv_half = (v_1.T.dot(np.diag(np.power(w_1, -0.5)))).dot(v_1)
        mean_lik = self.score_cov_inv_half.dot(generative_mean)
        self.generative_mean = np.squeeze(generative_mean)
        likelihood_loss = rr.signal_approximator(mean_lik, coef=1.)
        scaled_response_selector = rr.selector(~opt_vars,(2*p,), rr.affine_transform(self.score_cov_inv_half,
                                                                                       np.zeros(p)))

        self.likelihood_loss = rr.affine_smooth(likelihood_loss, scaled_response_selector)

        self.total_loss = rr.smooth_sum([self.randomization_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier,
                                         self.cube_barrier])

        self.p = p
        self.nactive = nactive
        self.cov_data_inv = np.linalg.inv(score_cov)
        self.observed_score_state = solver.observed_score_state

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

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current)
            ninactive = self.p + self.nactive
            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                proposal_opt = proposal[self.p:]
                failing_cube = (proposal[ninactive:] > self.inactive_lagrange) + \
                               (proposal[ninactive:] < - self.inactive_lagrange)

                failing_sign = (proposal_opt[:self.nactive] < 0)
                failing = failing_cube.sum()+failing_sign.sum()

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

class map_credible_split(selection_probability_split):

    def __init__(self, solver, prior_variance, coef=1., offset=None, quadratic=None):

        self.solver = solver
        X, _ = self.solver.loss.data
        self.p_shape = X.shape[1]
        self.param_shape = self.solver._overall.sum()
        self.prior_variance = prior_variance

        initial = solver.observed_opt_state[:self.param_shape]
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
        generative_mean[:self.param_shape]= sel_param

        sel_split = selection_probability_split(self.solver, generative_mean)

        cov_data_inv = sel_split.cov_data_inv

        sel_prob_primal = sel_split.minimize2(nstep=100)[::-1]

        optimal_primal = (sel_prob_primal[1])[:self.p_shape]

        sel_prob_val = -sel_prob_primal[0]

        full_gradient = cov_data_inv.dot(optimal_primal - generative_mean)

        optimizer = full_gradient[:self.param_shape]

        data_obs = sel_split.score_cov_inv_half.dot(sel_split.observed_score_state)

        likelihood_loss = rr.signal_approximator(data_obs, coef=1.)

        likelihood_loss = rr.affine_smooth(likelihood_loss, sel_split.score_cov_inv_half[:, :self.param_shape])

        likelihood_loss_value = likelihood_loss.smooth_objective(sel_param, 'func')

        likelihood_loss_grad = likelihood_loss.smooth_objective(sel_param, 'grad')

        log_prior_loss = rr.signal_approximator(np.zeros(self.param_shape), coef=1. /self.prior_variance)

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
                #print("proposal", proposal)

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

    def posterior_samples(self, Langevin_steps=2000, burnin=100):
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
            print i, sampler.state.copy()

        samples = np.array(samples)
        return samples[burnin:, :]













































