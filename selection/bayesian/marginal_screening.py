import numpy as np
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.selection_probability_rr import cube_subproblem_scaled, cube_barrier_scaled,\
    cube_gradient_scaled, cube_hessian_scaled, cube_objective
from selection.bayesian.barrier import barrier_conjugate_softmax_scaled_rr
import regreg.api as rr

class selection_probability_objective_ms(rr.smooth_atom):
    def __init__(self,
                 active,
                 active_signs,
                 threshold, # a vector in R^p
                 mean_parameter,  # in R^p
                 noise_variance,
                 randomizer,
                 epsilon = 0.,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        """
        Objective function for $\beta_E$ (i.e. active) with $E$ the `active_set` optimization
        variables, and data $z \in \mathbb{R}^n$ (i.e. response).
        NEEDS UPDATING
        Above, $\beta_E^*$ is the `parameter`, $b_{\geq}$ is the softmax of the non-negative constraint,
        $$
        B_E = X^TX_E
        $$
        and
        $$
        \gamma_E = \begin{pmatrix} \lambda s_E\\ 0\end{pmatrix}
        $$
        with $\lambda$ being `lagrange`.
        Parameters
        ----------
        X : np.float
             Design matrix of shape (n,p)
        active : np.bool
             Boolean indicator of active set of shape (p,).
        active_signs : np.float
             Signs of active coefficients, of shape (active.sum(),).
        lagrange : np.float
             Array of lagrange penalties for LASSO of shape (p,)
        parameter : np.float
             Parameter $\beta_E^*$ for which we want to
             approximate the selection probability.
             Has shape (active_set.sum(),)
        randomization : np.float
             Variance of IID Gaussian noise
             that was added before selection.
        """

        p = threshold.shape[0]
        E = active.sum()
        self.active = active

        #w, v = np.linalg.eig(noise_var_covar)
        #var_half_inv = (v.T.dot(np.diag(1./w))).dot(v)
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        #self.inactive_threshold = threshold[~active]

        initial = np.ones(p + E)

        rr.smooth_atom.__init__(self,
                                (p + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        nonnegative = nonnegative_softmax(E)

        opt_vars = np.zeros(p + E, bool)
        opt_vars[p:] = 1

        self._opt_selector = rr.selector(opt_vars, (p + E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (p + E,))

        self.A_active = np.hstack([-np.identity(E),np.zeros((E,p-E)),np.identity(E)*active_signs[None, :]])
        self.offset_active = active_signs * threshold[active]
        self.A_inactive = np.hstack([np.zeros((p-E,E)), np.identity(p-E), np.zeros((p-E,E))])

        # defines \gamma and likelihood loss
        self.set_parameter(mean_parameter, noise_variance)

        self.noise_variance = noise_variance

        self.active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                                 rr.affine_transform(self.A_active, self.offset_active))

        cube_obj = cube_objective(self.inactive_conjugate,
                                  threshold[~active],
                                  nstep=nstep)

        self.cube_loss = rr.affine_smooth(cube_obj, self.A_inactive)

        self.total_loss = rr.smooth_sum([self.active_conj_loss,
                                         self.cube_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier])

        self.threshold = threshold

    def set_parameter(self, mean_parameter, var_half_inv):
        """
        Set $\beta_E^*$.
        """
        mean_parameter = np.squeeze(mean_parameter)
        likelihood_loss = rr.signal_approximator(mean_parameter, coef=1.)
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
            f, g = self.total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize(self, initial=None, min_its=10, max_its=50, tol=1.e-10):

        nonneg_con = self._opt_selector.output_shape[0]
        constraint = rr.separable(self.shape,
                                  [rr.nonnegative((nonneg_con,), offset=1.e-12 * np.ones(nonneg_con))],
                                  [self._opt_selector.index_obj])

        problem = rr.separable_problem.fromatom(constraint, self)
        problem.coefs[:] = 0.5
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        value = problem.objective(soln)
        return soln, value

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        p = self.threshold.shape[0]
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
                if np.all(proposal[p:] > 0):
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

        #print('iter', itercount)
        value = objective(current)
        return current, value


class dual_selection_probability_ms(rr.smooth_atom):
    def __init__(self,
                 active,
                 active_signs,
                 threshold,  # a vector in R^p
                 mean_parameter,  # in R^p
                 noise_variance,
                 randomizer,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        p = threshold.shape[0]
        E = active.sum()

        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.CGF_randomization = randomizer.CGF

        if self.CGF_randomization is None:
            raise ValueError(
                'randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = -np.ones(p)

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        mean_parameter = np.squeeze(mean_parameter)

        self.active_slice = np.zeros_like(active, np.bool)
        self.active_slice[:active.sum()] = True

        self.B_active = np.hstack([np.identity(E) * active_signs[None, :], np.zeros((E, p - E))])
        self.B_inactive = np.hstack([np.zeros((p-E, E)), np.identity((p - E))])
        self.B_p = np.vstack((self.B_active, self.B_inactive))

        self.B_p_inv = np.linalg.inv(self.B_p.T)

        self.offset_active = active_signs * threshold[active]
        self.inactive_subgrad = np.zeros(p - E)

        self.cube_bool = np.zeros(p, np.bool)

        self.cube_bool[E:] = 1

        self.dual_arg = self.B_p_inv.dot(np.append(self.offset_active, self.inactive_subgrad))

        self._opt_selector = rr.selector(~self.cube_bool, (p,))

        self.set_parameter(mean_parameter, noise_variance)

        _barrier_star = barrier_conjugate_softmax_scaled_rr(self.cube_bool, threshold[~active])

        self.conjugate_barrier = rr.affine_smooth(_barrier_star, np.identity(p))

        self.CGF_randomizer = rr.affine_smooth(self.CGF_randomization, -self.B_p_inv)

        self.constant = np.true_divide(mean_parameter.dot(mean_parameter), 2 * noise_variance)

        self.linear_term = rr.identity_quadratic(0, 0, self.dual_arg, -self.constant)

        self.total_loss = rr.smooth_sum([self.conjugate_barrier,
                                         self.CGF_randomizer,
                                         self.likelihood_loss])

        self.total_loss.quadratic = self.linear_term

        self.threshold = threshold

    def set_parameter(self, mean_parameter, noise_variance):

        #mean_parameter = np.append(mean_parameter[self.active], mean_parameter[~self.active])

        mean_parameter = np.squeeze(mean_parameter)

        self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)

        self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, self.B_p_inv)

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

        E = self.active.sum()
        p = self.threshold.shape[0]
        current = np.append(-np.ones(E),np.ones(p-E))
        current_value = np.inf

        objective = lambda u: self.total_loss.objective(u)
        grad = lambda u: self.total_loss.smooth_objective(u, 'grad') + self.dual_arg

        for itercount in range(nstep):
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



