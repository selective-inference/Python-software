"""
This module contains a class `logistic`_ that implements
post selection inference for logistic LASSO.

"""

import warnings
from copy import copy

import numpy as np

from regreg.api import logistic_loss, weighted_l1norm, simple_problem, identity_quadratic

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

    randomization_sd = 0.1

    def __init__(self, y, X, weights,
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

        weights : np.float(p) or float
            Coefficients in weighted L-1 penalty in
            optimization problem. If a float,
            weights are proportional to 1.

        trials : np.int(n) (optional)
            Number of trials for each of the proportions.
            If not specified, defaults to np.ones(n)

        """
        self.y, self.X = y, X
        n, p = X.shape

        if np.array(weights).shape == ():
            weights = weights * np.ones(p)
        self.weights = weights

        if trials is None:
            trials = np.ones_like(y)
        self.trials = trials
        self._successes = self.trials * self.y
        n, p = X.shape

        if randomization_covariance is None:
            randomization_covariance = np.identity(p) * self.randomization_sd**2
        self.randomization_covariance = randomization_covariance
        self.randomization_chol = np.linalg.cholesky(self.randomization_covariance)
        self.random_linear_term = np.dot(self.randomization_chol,
                                         np.random.standard_normal(p))

    def fit(self, solve_args={'min_its':200, 'tol':1.e-10}):
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
        self.loss = logistic_loss(self.X, self.y * self.trials, trials=self.trials,
                                  coef=n/2.)
        self.penalty = weighted_l1norm(self.weights, lagrange=1.)
        self.penalty.quadratic = identity_quadratic(0, 0, self.random_linear_term, 0)
        problem = simple_problem(self.loss, self.penalty)
        soln = problem.solve(**solve_args)

        self._soln = soln
        if not np.all(soln == 0):
            self.active = np.nonzero(soln)[0]
            self.inactive = np.array(sorted(set(xrange(p)).difference(self.active)))
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

        self.z_E = np.sign(beta[self.active])
        X_E = self.X[:,self.active]

        self._linpred_unpenalized = np.dot(X_E, self._beta_unpenalized)
        self._fitted_unpenalized = p_unpen = np.exp(self._linpred_unpenalized) / (1. + np.exp(self._linpred_unpenalized))
        self._resid_unpenalized = self.y - self._fitted_unpenalized
        self._weight_unpenalized = p_unpen * (1 - p_unpen)

        # estimate covariance of (\bar{\beta}_E, X_{-E}^T(y-\pi_E(\bar{\beta}_E))
        # \approx (\bar{\beta}_E, X_{-E}^T(y-\pi_E(\beta_E^*)) - C_E(\beta_E^*) (\bar{\beta}_E - \beta_E^*)
        # \approx (\beta_E^* + Q_E(\beta^*_E)^{-1} X_E^T(y - \pi_E(\beta^*_E), X_{-E}^T(y-\pi_E(\beta_E^*)) - I_E(\beta_E^*) X_E^T(y - \pi_E(\beta^*_E))

        Q_Ei = self.quadratic_form_inv # plugin estimate of Q_E(\beta_E^*)^{-1}
        I_E = self._irrepresentable # plugin estimate of C_E(\beta_E^*) Q_E(\beta_E^*)^{-1}

        transform = np.zeros((p, p))
        transform[:s,self.active] = Q_Ei
        transform[s:,self.inactive] = np.identity(p-s)
        transform[s:,self.active] = -I_E

        # this is the covariance used in the constraint

        cov_asymp = np.zeros((2*p, 2*p))
        cov_asymp[:p,:p] = np.dot(transform, np.dot(self.covariance, transform.T))
        cov_asymp[p:,p:] = np.dot(self.randomization_chol,
                                  self.randomization_chol.T)

        linear_part = np.zeros((s + 2*(p-s), 2*p))
        linear_part[:s,:s] = -np.diag(self.z_E) 
        linear_part[:s,p+self.active] = self.z_E[:,None] * Q_Ei

        linear_part[s:p,s:p] = np.identity(p-s)
        linear_part[s:p,p+self.active] = I_E
        linear_part[s:p,p+self.inactive] = -np.identity(p-s)

        linear_part[p:] = -linear_part[s:p]

        offset = np.hstack([-self.z_E * np.dot(Q_Ei, self.z_E * self.weights[self.active]),
                             self.weights[self.inactive] - np.dot(I_E, self.z_E * self.weights[self.active]),
                             self.weights[self.inactive] + np.dot(I_E, self.z_E * self.weights[self.active])])

        self._constraints = constraints(linear_part,
                                        offset,
                                        covariance=cov_asymp)

        DEBUG = False
        if DEBUG: 
            # check KKT

            # compare to our linearized estimate of inactive gradient

            grad = self.loss.smooth_objective(self.soln, mode='grad') + self.random_linear_term
            print grad[self.active] + self.z_E * self.weights[self.active] , 'should be about 0'
            print np.fabs(grad[self.inactive] / self.weights[self.inactive]).max(), 'should be less than 1'

            approx_grad = (- np.dot(self.X[:,self.inactive].T, self._resid_unpenalized) + self.random_linear_term[self.inactive] 
                           - np.dot(I_E, self.z_E * self.weights[self.active] + self.random_linear_term[self.active]))

            import matplotlib.pyplot as plt
            plt.clf()
            plt.scatter(grad[self.inactive], approx_grad)
            print (np.linalg.norm(grad[self.inactive] - approx_grad) / np.linalg.norm(grad[self.inactive]))

           # not always feasible! 
            print np.linalg.norm(self._feasible[p+self.active] - self.random_linear_term[self.active]) / np.linalg.norm(self.random_linear_term[self.active])

            print (np.dot(self._constraints.linear_part[:s], self._feasible) - (-self.z_E * (self._beta_unpenalized - np.dot(Q_Ei , self.random_linear_term[self.active])))), 'check1'
            print (np.dot(self._constraints.linear_part[:s], self._feasible) - offset[:s]).max(), 'check2'
            print (np.dot(self._constraints.linear_part[s:p], self._feasible) - offset[s:p]), 'check3'
            #print (-np.dot(self._constraints.linear_part, self._feasible)[s:p] + self._constraints.offset[s:p] + approx_grad)

        self._feasible = np.hstack([self._beta_unpenalized,
                                    np.dot(self.X[:,self.inactive].T, self._resid_unpenalized),
                                    self.random_linear_term])

        if not self._constraints(self._feasible):
            warnings.warn('initial point not feasible for constraints')
        print np.linalg.norm(cov_asymp[:s,:s] - Q_Ei) / np.linalg.norm(Q_Ei)

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
            self._irrepresentable = np.dot(X_notE.T, self._weight_unpenalized[:, None] * np.dot(X_E, self._quad_form_inv))
        return self._quad_form

    @property
    def quadratic_form_inv(self, doc="Inverse of quadratic form in active block of KKT conditions."):
        if not hasattr(self, "_quad_form_inv"):
            self.quadratic_form
        return self._quad_form_inv

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def constraints(self):
        """
        Affine constraints for this LASSO problem.
        This is `self.active_constraints` stacked with
        `self.inactive_constraints`.
        """
        return self._constraints

    #@property
    def intervals(self):
        """
        Intervals for OLS parameters of active variables
        adjusted for selection.

        """
        raise NotImplementedError
#         if not hasattr(self, "_intervals"):
#             self._intervals = []
#             C = self.active_constraints
#             XEinv = self._XEinv
#             if XEinv is not None:
#                 for i in range(XEinv.shape[0]):
#                     eta = XEinv[i]
#                     _interval = C.interval(eta, self.y,
#                                            alpha=self.alpha,
#                                            UMAU=self.UMAU)
#                     self._intervals.append((self.active[i],
#                                             _interval[0], _interval[1]))
#             self._intervals = np.array(self._intervals, 
#                                        np.dtype([('index', np.int),
#                                                  ('lower', np.float),
#                                                  ('upper', np.float)]))
#         return self._intervals

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

if __name__ == "__main__":
    from selection.algorithms.lasso import instance
    n, p = 2000, 40
    X, y = instance(n=n,p=p)[:2]
    n, p = X.shape
    print n, p
    ybin = np.random.binomial(1, 0.5, size=(n,))
    # lasso.randomization_sd = 1.e-6
    L = lasso(ybin, X, 0.007 * np.sqrt(n))
    print 'lagrange', 0.005 * np.sqrt(n)
    L.fit()
