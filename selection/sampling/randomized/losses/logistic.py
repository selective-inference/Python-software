import numpy as np
from base import selective_loss
from regreg.smooth.glm import logistic_loss

class logistic_Xrandom(selective_loss):

    def __init__(self, X, y, 
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None):
        selective_loss.__init__(self, X.shape[1],
                                coef=coef,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.X = X.copy()
        self.y = y.copy()
        self._restricted_grad_beta = np.zeros(self.shape)
        self._loss = logistic_loss(self.X, self.y, coef=self.X.shape[0]/2.)

    def smooth_objective(self, beta, mode='both',
                         check_feasibility=False):

        return self._loss.smooth_objective(beta, mode=mode, check_feasibility=check_feasibility)

    # this is something that regreg does not know about, i.e.
    # what is data and what is not...

    def fit_E(self, active, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression after seeing the active set, without penalty. 
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.
        
        Parameters:
        ----------
        active: the active set from fitting the logistic lasso

        solve_args: passed to regreg.simple_problem.solve

        Returns:
        --------
        Set self._beta_unpenalized which will be used in the covariance matrix calculation.
        """

        self.active = active
        if self.active.any():
            self.inactive = ~active
            X_E = self.X[:, self.active] 
            loss_E = logistic_loss(X_E, self.y)
            self._beta_unpenalized = loss_E.solve(**solve_args)
            self.bootstrap_covariance()
        else:
            raise ValueError("Empty active set.")


    def bootstrap_covariance(self):
        """
        Bootstrap the covariance matrix of the sufficient statistic $X^T y$,
        through the use of the restricted unpenalized solution to the 
        problem $\bar{beta}_E$.
        
        Set the "_cov" field to be the bootstrapped covariance matrix.
        """
        if not hasattr(self, "_beta_unpenalized"):
            raise ValueError("method fit_E has to be called before computing the covariance")

        if not hasattr(self, "_cov"):

            # nonparametric bootstrap for covariance of X^Ty

            X, y = self.X, self.y
            n, p = X.shape
            nsample = 2000

            def pi(X):
                w = np.exp(np.dot(X[:,self.active], self._beta_unpenalized))
                return w / (1 + w)

            _mean_cum = 0
            self._cov = np.zeros((p, p))
            
            for _ in range(nsample):
                indices = np.random.choice(n, size=(n,), replace=True)
                y_star = y[indices]
                X_star = X[indices]
                Z_star = np.dot(X_star.T, y_star - pi(X_star))
                _mean_cum += Z_star
                self._cov += np.multiply.outer(Z_star, Z_star)
            self._cov /= nsample
            _mean = _mean_cum / nsample
            self._cov -= np.multiply.outer(_mean, _mean)
            self._cov_pinv = np.linalg.pinv(self._cov)
            self.L = np.linalg.cholesky(self._cov)

    @property
    def covariance(self, doc="Covariance of sufficient statistic $X^Ty$."):
        if not hasattr(self, "_cov"):
            self.bootstrap_covariance()

        return self._cov

    def gradient(self, beta):
        """
        Gradient of smooth part restricted to active set
        """

        if not hasattr(self, "_cov"):
            self.bootstrap_covariance()

        g = -(data - np.dot(self._cov, beta)) 
        return g

    def hessian(self, beta):
        """
        hessian is constant in this case.
        """
        if not hasattr(self, "_cov"):
            self.bootstrap_covariance()

        return self._cov

    def setup_sampling(self, data, mean, linear_part, value):
        """
        Set up the sampling conditioning on the KKT constraints as well as
        the linear constraints C * data = d

        Parameters:
        ----------
        data: 
        The subject of the sampling. In this case the gradient of loss at 0.

        mean: \beta^0_E

        sigma: default to None in logistic lasso 

        linear_part: C

        value: d
        """

        self.accept_data = 0
        self.total_data = 0

        P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
        I = np.identity(linear_part.shape[1])

        self.data = data 
        self.mean = mean

        self.R = I - P
        self.P = P
        self.linear_part = linear_part


    def proposal(self, data):
        if not hasattr(self, "L"):
            self.bootstrap_covariance()

        n, p = self.X.shape
        stepsize = 1. / np.sqrt(p)
        #new = data + stepsize * np.dot(self.R, 
        #                               np.dot(self.L, np.random.standard_normal(p)))
        new = data + stepsize * np.dot(self.R, 
                                       np.random.standard_normal(p))
        log_transition_p = self.logpdf(new) - self.logpdf(data)
        return new, log_transition_p

    def logpdf(self, data):
        return -((data-self.mean) * self._cov_pinv.dot(data-self.mean)).sum() / 2

    def update_proposal(self, state, proposal, logpdf):
        pass
