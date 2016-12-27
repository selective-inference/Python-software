import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import cube_barrier_scaled, cube_gradient_scaled, cube_hessian_scaled
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.barrier_fs import linear_map, fs_conjugate, barrier_conjugate_fs_rr


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
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + E,))

        sign_1 = np.zeros((1,1))
        sign_1[0:,:] = active_sign_1
        sign_2 = np.zeros((1, 1))
        sign_2[0:, :] = active_sign_2
        Projection = (X[:,~active_1].dot(np.linalg.inv(X[:,~active_1].T.dot(X[:,~active_1])))).dot(X[:,~active_1].T)
        P_1 = np.identity(n) - Projection

        #print sign_array.shape, X[:, active].T.shape, X[:, ~active].T.shape, np.zeros(p-E).shape
        self.A_active_1 = np.hstack([-X[:, active_1].T, sign_1, np.zeros((1,1))])
        self.A_active_2 = np.hstack([-X[:, active_2].T.dot(P_1), np.zeros((1, 1)), sign_2])

        self.A_in_1 = np.hstack([-X[:, ~active_1].T, np.zeros((p-1,1)), np.zeros((p-1,1))])
        self.A_in_2 = np.hstack([np.zeros((n,1)).T, np.ones((1,1)), np.zeros((1,1))])
        self.A_inactive_1 = np.vstack([self.A_in_1, self.A_in_2])

        self.A_in2_1 = np.hstack([-X[:, ~active_2].T.dot(P_1), np.zeros((p - 2, 1)), np.zeros((p - 2, 1))])
        self.A_in2_2 = np.hstack([np.zeros((n, 1)).T, np.zeros((1, 1)), np.ones((1, 1))])
        self.A_inactive_2 = np.vstack([self.A_in2_1, self.A_in2_2])

        self.set_parameter(mean_parameter, noise_variance)

        self.active_conj_loss_1 = rr.affine_smooth(self.active_conjugate, self.A_active_1)
        self.active_conj_loss_2 = rr.affine_smooth(self.active_conjugate, self.A_active_2)

        cube_obj = cube_objective_fs_linear(self.inactive_conjugate)

        self.cube_loss = rr.affine_smooth(cube_obj, self.A_inactive)

        self.total_loss = rr.smooth_sum([self.active_conj_loss,
                                         self.cube_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier])
