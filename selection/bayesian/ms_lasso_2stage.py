import numpy as np
import regreg.api as rr
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.selection_probability_rr import cube_subproblem_scaled, cube_barrier_scaled,\
    cube_gradient_scaled, cube_hessian_scaled, cube_objective
from selection.bayesian.credible_intervals import projected_langevin


class selection_probability_objective_fs_2steps(rr.smooth_atom):
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

        rr.smooth_atom.__init__(self,
                                (n + E_1 + E_2,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial
        nonnegative = nonnegative_softmax(E_1 + E_2)
        opt_vars = np.zeros(n + E_1 + E_2, bool)
        opt_vars[n:] = 1

        self._opt_selector = rr.selector(opt_vars, (n + E_1 + E_2,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + E_1 + E_2,))

        arg_ms = np.zeros(self.n + E_1 + E_2, bool)
        arg_ms[:self.n + E_1] = 1
        arg_lasso = np.zeros(self.n + E_1, bool)
        arg_lasso[:self.n] = 1
        arg_lasso = np.append(arg_lasso, np.ones(E_2, bool))

        self.A_active_1 = np.hstack([np.true_divide(-X[:, active_1].T,noise_variance),np.identity(E_1)
                                   *active_signs_1[None, :] ])
        self.A_inactive_1 = np.hstack([np.true_divide(-X[:, ~active_1].T, noise_variance), np.zeros((p - E_1, E_1))])

        self.offset_active_1 = active_signs_1 * threshold[active_1]
        self.offset_inactive_1 = np.zeros(p - E_1)

        self._active_ms = rr.selector(arg_ms, (self.n + E_1 + E_2,),
                                         rr.affine_transform(self.A_active_1, self.offset_active_1))

        self._inactive_ms = rr.selector(arg_ms, (self.n + E_1 + E_2,),
                                           rr.affine_transform(self.A_inactive_1, self.offset_inactive_1))

        self.active_conj_loss_1 = rr.affine_smooth(self.active_conjugate, self._active_ms)

        cube_obj_1 = cube_objective(self.inactive_conjugate, lagrange[~active_2], nstep=nstep)

        self.cube_loss_1 = rr.affine_smooth(cube_obj_1, self._inactive_ms)

        X_E_2 = X[:, active_2]
        B = X.T.dot(X_E_2)

        B_E = B[active_2]
        B_mE = B[~active_2]

        self.A_active_2 = np.hstack([-X[:, active_2].T, (B_E + epsilon * np.identity(E_2)) * active_signs_2[None, :]])
        self.A_inactive_2 = np.hstack([-X[:, ~active_2].T, (B_mE * active_signs_2[None, :])])

        self.offset_active_2 = active_signs_2 * lagrange[active_2]

        self.offset_inactive_2 = np.zeros(p-E_2)


        self._active_lasso = rr.selector(arg_lasso, (self.n + E_1 + E_2,),
                                           rr.affine_transform(self.A_active_2, self.offset_active_2))

        self._inactive_lasso = rr.selector(arg_lasso, (self.n + E_1 + E_2,),
                                           rr.affine_transform(self.A_inactive_2, self.offset_inactive_2))

        self.active_conj_loss_2 = rr.affine_smooth(self.active_conjugate, self._active_lasso)

        cube_obj_2 = cube_objective(self.inactive_conjugate, lagrange[~active_2], nstep=nstep)

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











