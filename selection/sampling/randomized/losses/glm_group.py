import numpy as np
from regreg.smooth.glm import glm as regreg_glm
import regreg.api as rr

class group_lasso_sampler(object):

    def __init__(self, loss, initial_soln, epsilon, penalty, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression to a candidate active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.

        Computes $\bar{\beta}_E$ which is the restricted 
        MLE (i.e. subject to the constraint $\beta_{-E}=0$).

        Parameters:
        -----------

        active: np.bool
            The active set from fitting the logistic lasso

        solve_args: dict
            Arguments to be passed to regreg solver.

        Returns:
        --------

        None

        Notes:
        ------

        Sets self._beta_unpenalized which will be used in the covariance matrix calculation.
        Also computes Hessian of loss at restricted MLE as well as the bootstrap covariance.

        """

        # find the active groups and their direction vectors
        # as well as unpenalized groups

        groups = np.unique(penalty.groups) 
        active_groups = np.zeros(len(groups), np.bool)
        unpenalized_groups = np.zeros(len(groups), np.bool)

        active_directions = []
        active = np.zeros(loss.shape, np.bool)
        unpenalized = np.zeros(loss.shape, np.bool)
        for i, g in enumerate(groups):
            group = penalty.groups == g
            active_groups[i] = (np.linalg.norm(initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros_like(active)
                z[group] = initial_soln[group] / np.linalg.norm(initial_soln[group])
                active_directions.append(z)
            if unpenalized_groups[i]:
                unpenalized[group] = True

        active_directions = np.array(active_directions).T

        # solve the restricted problem

        X, Y = loss.data
        if self._is_transform:
            raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
        X_E = X[:,active]
        loss_E = rr.affine_smooth(self.loss, X_E)
        _beta_unpenalized = loss_E.solve(**solve_args)
        beta_full = np.zeros(active.shape)
        beta_full[active] = _beta_unpenalized
        _hessian = loss.hessian(beta_full)

        # form linear part

        inactive = ~active
        self.num_opt_var = p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _linear_term = np.zeros((p, p + active_groups.sum() + unpenalized.sum() + inactive.sum())) 

        # \bar{\beta}_{E \cup U} piece

        overall = active + unpenalized
        mle_slice = slice(0, overall.sum())
        _mle_hessian = _hessian[:,overall]
        _linear_term[:,mle_slice] = -_mle_hessian

        # N_{-E} piece

        null_slice = slice(overall.sum(), p)
        _linear_term[inactive][:,null_slice] = -np.identity(inactive.sum())

        # c_E piece 

        scaling_slice = slice(p, p + active_groups.sum())
        _opt_hessian = _hessian.dot(active_directions)
        _linear_term[:,scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(p + active_groups.sum(), p + active_groups.sum() + unpenalized.sum())
        _linear_term[:,unpenalized_slice] = _hessian[:,unpenalized]

        # subgrad piece
        subgrad_slice = slice(p + active_groups.sum() + unpenalized.sum(), p + active_groups.sum() + inactive.sum() + unpenalized.sum())
        _linear_term[inactive][:,subgrad_slice] = np.identity(inactive.sum())

        # form affine part

        _affine_term = np.zeros(p)
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _affine_term[group] = active_directions[i][group] * penalty.weights[g]

        # two transforms that encode data and optimization
        # variable roles 

        # later, conditioning will modify `data_transform`

        self.opt_transform = rr.affine_transform(_linear_term[:,p:], _affine_term)
        self.data_transform = rr.linear_transform(_linear_term[:,:p])

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables, so it will be assumed to working already only
        # on last p coordinates

        self.scaling_slice = scaling_slice[p:]

        new_groups = penalty.groups[inactive]
        new_weights = [penalty.weights[g] for g in penalty.weights.keys() if g in np.unique(new_groups)]

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice[p:]

    def projection(self, opt_state):
        """
        Full projection for Langevin.

        The state here will be only the state of the optimization variables.
        """

        new_state = state.copy() # not really necessary to copy
        new_state[self.scaling_slice] = np.maximum(state[self.scaling_slice], 0)
        new_state[self.subgrad_slice] = self.group_lasso_dual.bound_prox(state[self.subgrad_slice])

        return new_state

    def gradient(self, data_state, opt_state):
        """
        Randomization derivative at full state.
        """
        omega = full_state = (self.data_transform.affine_map(data_state) + 
                              self.opt_transform.affine_map(opt_state))
        randomization_derivative = self.randomization.gradient(omega)
        data_grad = self.data_transform.adjoint_map(randomization_derivative)
        opt_grad = self.opt_transform.adjoint_map(randomization_derivative)
        return data_grad, opt_grad

    def bootstrap_covariance(self, nsample = 2000):
        """
        """
        if not hasattr(self, "_beta_unpenalized"):
            raise ValueError("method fit_restricted has to be called before computing the covariance")

        if not hasattr(self, "_cov"):

            X, y = self.data

            if isinstance(self.loss, logistic_loglike):
                y = y[0]
            n, p = X.shape
            active=self.active
            inactive=~active

            beta_full = np.zeros(X.shape[1])
            beta_full[self.active] = self._beta_unpenalized

            def mu(X):
                return self.loss.smooth_objective(X.dot(beta_full), 'grad') + y 

            _mean_cum = 0

            self._cov = np.zeros((p,p))
            Q = np.zeros((p,p))

            W = np.diag(self.loss.hessian(X.dot(beta_full)))
            Q = np.dot(X[:, active].T, np.dot(W, X[:, active]))
            Q_inv = np.linalg.inv(Q)
            C = np.dot(X[:, inactive].T, np.dot(W, X[:, active]))
            I = np.dot(C,Q_inv)

            for _ in range(nsample):
                indices = np.random.choice(n, size=(n,), replace=True)
                y_star = y[indices]
                X_star = X[indices]
                mu_star = mu(X_star)

                Z_star_active = np.dot(X_star[:, active].T, y_star - mu_star)
                Z_star_inactive = np.dot(X_star[:, inactive].T, y_star - mu_star)

                Z_1 = np.dot(Q_inv, Z_star_active)
                Z_2 = Z_star_inactive + np.dot(I, Z_star_active)
                Z_star = np.concatenate((Z_1, Z_2), axis=0)

                _mean_cum += Z_star
                self._cov += np.multiply.outer(Z_star, Z_star)

            self._cov /= float(nsample)
            _mean = _mean_cum / float(nsample)
            self._cov -= np.multiply.outer(_mean, _mean)

            return self._cov

    @property
    def covariance(self, doc="Covariance of score $X^T(y - \mu(\beta_E^*))$."):
        if not hasattr(self, "_cov"):
            self.bootstrap_covariance()
        return self._cov





