
import numpy as np
import regreg.api as rr
from selection.algorithms.softmax import nonnegative_softmax
from selection.randomized.M_estimator import M_estimator_split
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov


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
        BIG = 10 ** 10
        _diff = arg - self.lagrange_cube  # z - \lambda < 0
        _sum = arg + self.lagrange_cube  # z + \lambda > 0
        violations = ((_diff >= 0).sum() + (_sum <= 0).sum() > 0)

        f = np.log((_diff - 1.) * (_sum + 1.) / (_diff * _sum)).sum() + BIG * violations
        g = 1. / (_diff - 1) - 1. / _diff + 1. / (_sum + 1) - 1. / _sum

        if mode == 'func':
            return self.scale(f)
        elif mode == 'grad':
            return self.scale(g)
        elif mode == 'both':
            return self.scale(f), self.scale(g)
        else:
            raise ValueError('mode incorrectly specified')


class selection_probability_split(rr.smooth_atom, M_estimator_split):

    def __init__(self, loss, epsilon, penalty, generative_mean, subsample_size, coef=1., offset=None, quadratic=None, nstep=10):

        M_estimator_split.__init__(self, loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10})

        self.Msolve()

        X, _ = self.loss.data
        n, p = X.shape
        nactive = self._overall.sum()

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        self.feasible_point = self.observed_opt_state

        initial = np.zeros(2*p, )
        initial[p:] = self.feasible_point

        rr.smooth_atom.__init__(self,
                                (2*p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        score_linear_term = self.score_transform[0]

        (opt_linear_term, opt_affine_term) = self.opt_transform

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

        randomization_cov = self.setup_sampler()
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

class sel_prob_gradient_map_split(rr.smooth_atom):

    def __init__(self, sel_prob, true_dim, coef=1., offset=None, quadratic=None):

        self.sel_prob = sel_prob
        self.p = self.sel_prob.generative_mean.shape[0]

        self.dim = true_dim

        self.cov_data_inv = np.linalg.inv(self.sel_prob.score_cov)

        rr.smooth_atom.__init__(self,(self.dim,),offset=offset,quadratic=quadratic,coef=coef)

    def smooth_objective(self, param, mode='both', check_feasibility=False, tol=1.e-6):

        mean_parameter = self.sel_prob.generative_mean

        sel_prob_primal = self.sel_prob.minimize2(nstep=100)[::-1]

        optimal_primal = (sel_prob_primal[1])[:self.p]

        sel_prob_val = -sel_prob_primal[0]

        full_gradient = self.cov_data_inv .dot(optimal_primal - mean_parameter)

        optimizer = full_gradient[:self.dim]

        if mode == 'func':
            return sel_prob_val
        elif mode == 'grad':
            return optimizer
        elif mode == 'both':
            return sel_prob_val, optimizer
        else:
            raise ValueError('mode incorrectly specified')































