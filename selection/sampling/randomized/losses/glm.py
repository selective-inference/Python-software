import numpy as np
from regreg.smooth.glm import glm as regreg_glm, logistic_loglike
import regreg.api as rr

class glm(regreg_glm):

    # this is something that regreg does not know about, i.e.
    # what is data and what is not...

    def fit_restricted(self, active, solve_args={'min_its':50, 'tol':1.e-10}):
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

        self.active = active
        size_active = np.sum(self.active)

        if self.active.any():
            self.inactive = ~active
            X, Y = self.data
            if self._is_transform:
                raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
            X_E = X[:, self.active] 
            loss_E = rr.affine_smooth(self.saturated_loss, X_E)
            self._beta_unpenalized = loss_E.solve(**solve_args)
            beta_full = np.zeros(active.shape)
            beta_full[active] = self._beta_unpenalized
            self._hessian = self.hessian(beta_full)
            _restricted_hessian = self._hessian[:, self.active]
            self._restricted_hessian = np.zeros_like(_restricted_hessian)
            self._restricted_hessian[:size_active] = _restricted_hessian[self.active]
            self._restricted_hessian[size_active:] = _restricted_hessian[self.inactive]
            self.bootstrap_covariance()
        else:
            raise ValueError("Empty active set.")

    def bootstrap_covariance(self, nsample = 2000):
        """
        """
        if not hasattr(self, "_beta_unpenalized"):
            raise ValueError("method fit_restricted has to be called before computing the covariance")

        if not hasattr(self, "_cov"):

            X, y = self.data

            n, p = X.shape
            active=self.active
            inactive=~active

            beta_full = np.zeros(X.shape[1])
            beta_full[self.active] = self._beta_unpenalized

            def mu(X):
                return self.saturated_loss.smooth_objective(X.dot(beta_full), 'grad') + y 

            _mean_cum = 0

            self._cov = np.zeros((p,p))
            Q = np.zeros((p,p))

            W = np.diag(self.saturated_loss.hessian(X.dot(beta_full)))
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





