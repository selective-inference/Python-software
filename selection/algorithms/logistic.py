"""
This module contains a class `logistic`_ that implements
post selection inference for logistic LASSO.

"""

import warnings
from copy import copy

import numpy as np

from regreg.api import logistic_loss, l1norm, simple_problem, identity_quadratic

from selection.constraints.affine import (constraints, selection_interval,
                                 interval_constraints,
                                 sample_from_constraints,
                                 gibbs_test,
                                 stack)
from selection.distributions.discrete_family import discrete_family
from selection.algorithms.lasso import lasso as OLS_lasso, _constraint_from_data

class lasso(OLS_lasso):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta;X,Y) + 
             \lambda \|\beta\|_1

    where $\lambda$ is `lam` and $\ell$ is logistic loss.


    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    randomization_sd = 0.001

    def __init__(self, y, X, lam,
                 trials=None,
                 randomization_covariance=None):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        y : np.float(n)
            The response vector assumed to be a proportion, with
            each entry based on Binomial with a given number
            of trials.

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        lam : np.float
            Coefficient of the L-1 penalty in
            $\text{minimize}_{\beta} \frac{1}{2} \|y-X\beta\|^2_2 + 
                \lambda\|\beta\|_1$

        trials : np.int(n) (optional)
            Number of trials for each of the proportions.
            If not specified, defaults to np.ones(n)

        """
        self.y, self.X = y, X
        n, p = X.shape
        if trials is None:
            trials = np.ones_like(y)
        self.trials = trials
        self._successes = self.trials * self.y
        n, p = X.shape
        self.lagrange = lam

        if randomization_covariance is None:
            randomization_covariance = np.identity(p) * self.randomization_sd**2
        self.randomization_covariance = randomization_covariance
        self.randomization_chol = np.linalg.cholesky(self.randomization_covariance)
        self.random_linear_term = np.dot(self.randomization_chol,
                                         np.random.standard_normal(p))

    def fit(self, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fit the lasso using `regreg`.
        This sets the attribute `soln` and
        forms the constraints necessary for post-selection inference
        by calling `form_constraints()`.

        Parameters
        ----------

        solve_args : {}
             Passed to `regreg.simple_problem.solve``_

        Returns
        -------

        soln : np.float
             Solution to lasso with `sklearn_alpha=self.lagrange`.
             
        """

        n, p = self.X.shape
        loss = logistic_loss(self.X, self.y * self.trials, trials=self.trials)
        penalty = l1norm(p, lagrange=self.lagrange)
        penalty.quadratic = identity_quadratic(0, 0, self.random_linear_term, 0)
        problem = simple_problem(loss, penalty)
        soln = problem.solve(**solve_args)

        self._soln = soln
        if not np.all(soln == 0):
            self.active = np.nonzero(soln)[0]
            self.inactive = sorted(set(xrange(p)).difference(self.active))
            X_E = self.X[:,self.active]
            loss_E = logistic_loss(X_E, self.y * self.trials, trials=self.trials)
            self._beta_unpenalized = loss_E.solve(**solve_args)
            self.form_constraints()
        else:
            self.active = []

    def form_constraints(self):
        """
        After having fit lasso, form the constraints
        necessary for inference.

        This sets the attributes: `active_constraints`,
        `inactive_constraints`, `active`.

        Returns
        -------

        None

        """

        # determine equicorrelation set and signs
        X, y = self.X, self. y
        beta = self.soln
        n, p = X.shape
        sparsity = s = self.active.shape[0]
        lam = self.lagrange 

        self.z_E = np.sign(beta[self.active])
        X_E = self.X[:,self.active]

        self._linpred_unpenalized = np.dot(X_E, self._beta_unpenalized)
        self._fitted_unpenalized = p_unpen = np.exp(self._linpred_unpenalized) / (1. + np.exp(self._linpred_unpenalized))
        self._weight_unpenalized = p_unpen * (1 - p_unpen)

        active_selector = np.identity(p)[self.active]
        active_linear = -self.quadratic_form
        active_offset = - lam * self.z_E
        self._active_constraints = constraints(np.dot(active_linear, active_selector),
                                               active_offset,
                                               covariance=self.covariance)
                                               
        inactive_linear = np.zeros((p-s, p))
        inactive_linear[:,self.inactive] = np.identity(p-s)
        inactive_linear[:,self.active] = -self._irrepresentable
        inactive_offset = np.ones(2*(p-s)) * lam
        inactive_offset[:(p-s)] += -lam * np.dot(self._irrepresentable, self.z_E)
        inactive_offset[:(p-s)] += lam * np.dot(self._irrepresentable, self.z_E)
        self._inactive_constraints = constraints(np.vstack([inactive_linear,
                                                            -inactive_linear]),
                                                 inactive_offset,
                                                 covariance=self.covariance)
                                               
        self._constraints = stack(self._active_constraints,
                                  self._inactive_constraints)

        # NOTE: our actual Gaussian will typically not satisfy these constraints
        print self._constraints(np.dot(self.X.T, self.y - self._fitted_unpenalized))

    @property
    def covariance(self, doc="Covariance of sufficient statistic $X^Ty$."):
        if not hasattr(self, "_cov"):

            # nonparametric bootstrap for covariance of X^Ty

            X, y = self.X, self. y
            nsample = 2000

            boot_pop_mean = np.dot(X.T, y)
            self._cov = np.zeros((p, p))
            for _ in range(nsample):
                indices = np.random.choice(n, size=(n,), replace=True)
                y_star = y[indices]
                X_star = X[indices]
                Z_star = np.dot(X_star.T, y_star)
                self._cov += np.multiply.outer(Z_star, Z_star)
            self._cov /= nsample
            self._cov -= np.multiply.outer(boot_pop_mean, boot_pop_mean)
        return self._cov

    @property
    def quadratic_form(self, doc="Quadratic form in active block of KKT conditions."):
        if not hasattr(self, "_quad_form"):
            beta = self.soln
            X_E = self.X[:,self.active]
            X_notE = self.X[:,self.inactive]
            self._quad_form = np.dot(X_E.T * self._weight_unpenalized[None, :], X_E)
            self._quad_form_inv = np.linalg.pinv(self._quad_form)
            self._irrepresentable = np.dot(X_notE.T, np.dot(X_E, self._quad_form_inv))
        return self._quad_form

    @property
    def quadratic_form_inv(self, doc="Inverse of quadratic form in active block of KKT conditions."):
        if not hasattr(self, "_quad_form_inv"):
            self.quadratic_form
        return self._quad_form_inv

    @property
    def sampling_covariance(self, doc="Covariance of $X^Ty$ and its randomized version."):
        if not hasattr(self, "_sampling_cov"):
            n, p = self.X.shape
            cov = np.zeros((2*p, 2*p))
            cov[:p,:p] = self.covariance
            cov[:p,p:] = cov[p:,:p] = self.covariance
            cov[p:,p:] = self.covariance + self.randomization_covariance
            self._sampling_cov = cov
        return self._sampling_cov

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def active_constraints(self):
        """
        Affine constraints imposed on the
        active variables by the KKT conditions.
        """
        return self._active_constraints

    @property
    def inactive_constraints(self):
        """
        Affine constraints imposed on the
        inactive subgradient by the KKT conditions.
        """
        return self._inactive_constraints

    @property
    def constraints(self):
        """
        Affine constraints for this LASSO problem.
        This is `self.active_constraints` stacked with
        `self.inactive_constraints`.
        """
        return self._constraints

    @property
    def intervals(self):
        """
        Intervals for OLS parameters of active variables
        adjusted for selection.

        """
        if not hasattr(self, "_intervals"):
            self._intervals = []
            C = self.active_constraints
            XEinv = self._XEinv
            if XEinv is not None:
                for i in range(XEinv.shape[0]):
                    eta = XEinv[i]
                    _interval = C.interval(eta, self.y,
                                           alpha=self.alpha,
                                           UMAU=self.UMAU)
                    self._intervals.append((self.active[i],
                                            _interval[0], _interval[1]))
            self._intervals = np.array(self._intervals, 
                                       np.dtype([('index', np.int),
                                                 ('lower', np.float),
                                                 ('upper', np.float)]))
        return self._intervals

    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
        " for selection."):
        if not hasattr(self, "_pvals"):
            self._pvals = []
            C = self.active_constraints
            XEinv = self._XEinv
            if XEinv is not None:
                for i in range(XEinv.shape[0]):
                    eta = XEinv[i]
                    _pval = C.pivot(eta, self.y)
                    _pval = 2 * min(_pval, 1 - _pval)
                    self._pvals.append((self.active[i], _pval))
        return self._pvals

    @property
    def nominal_intervals(self):
        """
        Intervals for OLS parameters of active variables
        that have not been adjusted for selection.
        """
        if not hasattr(self, "_intervals_unadjusted"):
            if not hasattr(self, "_constraints"):
                self.form_constraints()
            self._intervals_unadjusted = []
            XEinv = self._XEinv
            SigmaE = self.sigma**2 * np.dot(XEinv, XEinv.T)
            for i in range(self.active.shape[0]):
                eta = XEinv[i]
                center = (eta*self.y).sum()
                width = ndist.ppf(1-self.alpha/2.) * np.sqrt(SigmaE[i,i])
                _interval = [center-width, center+width]
                self._intervals_unadjusted.append((self.active[i], eta, (eta*self.y).sum(), 
                                        _interval))
        return self._intervals_unadjusted

if __name__ == "__main__":
    from selection.algorithms.lasso import instance
    X, y = instance(n=2000)[:2]
    X = X[:,:50]
    n, p = X.shape
    ybin = np.random.binomial(1, 0.5, size=(n,))
    L = lasso(ybin, X, 0.001)
    L.fit()
