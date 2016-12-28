import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import cube_barrier_scaled, cube_gradient_scaled, cube_hessian_scaled
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.barrier_fs import linear_map, fs_conjugate, barrier_conjugate_fs_rr
from selection.bayesian.forward_step_reparametrized import cube_subproblem_fs_linear, cube_objective_fs_linear


class selection_probability_objective_fs_2steps(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,
                 active_1,
                 active_2,
                 active_sign_1,
                 active_sign_2,
                 mean_parameter,  # in R^n
                 noise_variance,
                 randomizer,
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

        n, p = X.shape
        E = 2
        self._X = X
        X_step2 = X[:,~active_1]
        self.active_1 = active_1
        self.active_2 = active_2
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = np.zeros(n + E, )
        initial[n:] = feasible_point

        rr.smooth_atom.__init__(self,
                                (n + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        nonnegative = nonnegative_softmax(E)

        opt_vars = np.zeros(n + E, bool)
        opt_vars[n:] = 1

        self._opt_selector = rr.selector(opt_vars, (n + E,))

        arg_inactive_loss = np.zeros(n + E, bool)
        arg_inactive_loss_1 = arg_inactive_loss
        arg_inactive_loss_1[:n+1] = 1
        arg_inactive_loss_2 = arg_inactive_loss
        arg_inactive_loss_2[:n] = 1
        arg_inactive_loss_2[n+1] = 1


        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + E,))

        sign_1 = np.zeros((1,1))
        sign_1[0:,:] = active_sign_1
        sign_2 = np.zeros((1, 1))
        sign_2[0:, :] = active_sign_2
        Projection = (X_step2.dot(np.linalg.inv(X_step2.T.dot(X_step2)))).dot(X_step2.T)
        P_1 = np.identity(n) - Projection

        #print sign_array.shape, X[:, active].T.shape, X[:, ~active].T.shape, np.zeros(p-E).shape
        self.A_active_1 = np.hstack([-X[:, active_1].T, sign_1, np.zeros((1,1))])
        self.A_active_2 = np.hstack([-X_step2[:, active_2].T.dot(P_1), np.zeros((1, 1)), sign_2])

        self.A_in_1 = np.hstack([-X[:, ~active_1].T, np.zeros((p-1,1))])
        self.A_in_2 = np.hstack([np.zeros((n,1)).T, np.ones((1,1))])
        self.A_inactive_1 = np.vstack([self.A_in_1, self.A_in_2])

        self.A_in2_1 = np.hstack([-X_step2[:, ~active_2].T.dot(P_1), np.zeros((p - 2, 1))])
        self.A_in2_2 = np.hstack([np.zeros((n, 1)).T, np.ones((1, 1))])
        self.A_inactive_2 = np.vstack([self.A_in2_1, self.A_in2_2])

        self.A_inactive_sel_1 = rr.selector(arg_inactive_loss_1, (n + E,),
                                            rr.affine_transform(self.A_inactive_1, np.zeros(n + 1)))

        self.A_inactive_sel_2 = rr.selector(arg_inactive_loss_2, (n + E,),
                                            rr.affine_transform(self.A_inactive_2, np.zeros(n + 1)))

        self.set_parameter(mean_parameter, noise_variance)

        self.active_conj_loss_1 = rr.affine_smooth(self.active_conjugate, self.A_active_1)
        self.active_conj_loss_2 = rr.affine_smooth(self.active_conjugate, self.A_active_2)

        cube_obj = cube_objective_fs_linear(self.inactive_conjugate)

        self.cube_loss_1 = rr.affine_smooth(cube_obj, self.A_inactive_sel_1)
        self.cube_loss_2 = rr.affine_smooth(cube_obj, self.A_inactive_sel_2)

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
            f, g = self.total_loss.smooth_objective(param, 'both')
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



