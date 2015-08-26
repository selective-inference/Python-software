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

    randomization_sd = 0.3

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

        self.statistical_func = np.hstack([self._beta_unpenalized,
                                           np.dot(self.X[:,self.inactive].T, self._resid_unpenalized),
                                           self.random_linear_term])

        self.feasible = np.zeros(2*p)
        self.feasible[p+self.active] = - self.z_E * self.weights[self.active]
        self.feasible[:s] = np.fabs(self._beta_unpenalized) * self.z_E
        self.feasible[s:p] = np.dot(self.X[:,self.inactive].T, self._resid_unpenalized)
        self.feasible[p+self.inactive] = self.feasible[s:p] + (np.random.sample(self.inactive.shape) * 2 * self.weights[self.inactive] - self.weights[self.inactive])

        if not self._constraints(self.feasible):
            warnings.warn('initial point not feasible for constraints')

    @property
    def covariance(self, doc="Covariance of sufficient statistic $X^Ty$."):
        if not hasattr(self, "_cov"):

            # nonparametric bootstrap for covariance of X^Ty

            X, y = self.X, self. y
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

    def hypothesis_test(self, linear_func, null_value=0,
                        alternative='twosided',
                        saturated=True,
                        ndraw=8000,
                        burnin=2000):
        """
        Test a null hypothesis of the form

        .. math::

             H_0: \eta^T\beta_E = 0

        in the selected model.

        Parameters
        ----------

        linear_func : np.float(*)
              Linear functional of shape self.active.shape

        null_value : float
              Specified value under null hypothesis

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        saturated : bool
            Should we assume that expected value
            of inactive gradient is 0 or not? If True,
            the test conditions on the asymptotically
            Gaussian $X_{-E}^T(y - \pi_E(\bar{\beta}_E))$.
            One "selected" model assumes this has mean 0.
            The saturated model does not make this assumption.

        ndraw : int (optional)
            Defaults to 8000.

        burnin : int (optional)
            Defaults to 2000.

        Returns
        -------

        pvalue : np.float
              P-value based on specified alternative.

        """

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        n, p = self.X.shape
        sparsity = s = self.active.shape[0]
        null_basis = np.random.standard_normal((s-1,s))
        normsq_linear_func = (linear_func**2).sum()
        for j in range(s-1):
            null_basis[j] -= (null_basis[j] * linear_func) * linear_func / normsq_linear_func

        if saturated:
            conditioning_func = np.zeros((p-1, 2*p))
            conditioning_func[:s-1,:s] = null_basis
            conditioning_func[(s-1):,s:p] = np.identity(p-s)
        else:
            conditioning_func = np.zeros((s-1, 2*p))
            conditioning_func[:s-1,:s] = null_basis

        expanded_linear_func = np.zeros(2*p)
        expanded_linear_func[:s] = linear_func

        conditioning_func = np.dot(conditioning_func, np.linalg.pinv(self.constraints.covariance))

        conditional_con = self.constraints.conditional(conditioning_func,
                                                       np.dot(conditioning_func, self.statistical_func))
        if not self.constraints(self.feasible):
            raise ValueError('need a feasible point for sampling')

        _, _, _, family = gibbs_test(conditional_con,
                                     self.feasible,
                                     expanded_linear_func,
                                     UMPU=False,
                                     sigma_known=True,
                                     ndraw=ndraw,
                                     burnin=burnin,
                                     alternative=alternative)

        observed = (expanded_linear_func * self.statistical_func).sum()
        if alternative == 'greater':
            pval = family.ccdf(0, observed)
        elif alternative == 'less':
            pval = family.cdf(0, observed)
        else:
            pval = family.cdf(0, observed)
            pval = 2 * min(pval, 1 - pval)
        return pval

if __name__ == "__main__":
    from selection.algorithms.lasso import instance

    def simulate(n=100, p=20):

        X, y = instance(n=n,p=p)[:2]
        n, p = X.shape

        ybin = np.random.binomial(1, 0.5, size=(n,))
        L = lasso(ybin, X, 0.03 * np.sqrt(n), 
                  randomization_covariance = 0.2 * np.dot(X.T, X) / 4.)
        L.fit()
        print L.active.shape
        v = np.ones_like(L.active)
        return L.hypothesis_test(v), L
