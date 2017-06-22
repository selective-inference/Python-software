"""
This module contains a class `lasso`_ that implements
post selection for the lasso
as described in `post selection LASSO`_.


.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238
.. _sample carving: http://arxiv.org/abs/1410.2597

"""

from __future__ import division

import warnings, functools
from copy import copy

import numpy as np
import pandas as pd
from scipy.stats import norm as ndist, t as tdist
from scipy.linalg import block_diag

from regreg.api import (glm, 
                        weighted_l1norm, 
                        simple_problem,
                        coxph as coxph_obj,
                        smooth_sum)

from .sqrt_lasso import solve_sqrt_lasso, estimate_sigma

from ..constraints.affine import (constraints, selection_interval,
                                 interval_constraints,
                                 sample_from_constraints,
                                 gibbs_test,
                                 stack)

from ..distributions.discrete_family import discrete_family
from ..randomized.glm import pairs_bootstrap_glm

class lasso(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \frac{1}{2n} \|y-X\beta\|^2_2 + 
            \lambda \|\beta\|_1

    where $\lambda$ is `lam`.

    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    def __init__(self, loglike, 
                 feature_weights,
                 covariance_estimator=None,
                 ignore_inactive_constraints=False):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        feature_weights : np.ndarray
            Feature weights for L-1 penalty. If a float,
            it is brodcast to all features.

        covariance_estimator : callable (optional)
            If None, use the parameteric
            covariance estimate of the selected model.

        Notes
        -----

        If not None, `covariance_estimator` should 
        take arguments (beta, active, inactive)
        and return an estimate of the covariance of
        $(\bar{\beta}_E, \nabla \ell(\bar{\beta}_E)_{-E})$,
        the unpenalized estimator and the inactive
        coordinates of the gradient of the likelihood at
        the unpenalized estimator.

        """

        self.loglike = loglike
        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        self.covariance_estimator = covariance_estimator
        self.ignore_inactive_constraints = ignore_inactive_constraints

    def fit(self, lasso_solution=None, solve_args={'tol':1.e-12, 'min_its':50}):
        """
        Fit the lasso using `regreg`.
        This sets the attributes `soln`, `onestep` and
        forms the constraints necessary for post-selection inference
        by calling `form_constraints()`.

        Parameters
        ----------

        lasso_solution : optional

             If not None, this is taken to be the solution
             of the optimization problem. No checks
             are done, though the implied affine
             constraints will generally not be satisfied.

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        Returns
        -------

        soln : np.float
             Solution to lasso.
             
        Notes
        -----

        If `self` already has an attribute `lasso_solution`
        this will be taken to be the solution and 
        no optimization problem will be solved. Supplying
        the optional argument `lasso_solution` will
        overwrite `self`'s `lasso_solution`.

        """

        self._penalty = weighted_l1norm(self.feature_weights, lagrange=1.)
        if lasso_solution is None and not hasattr(self, "lasso_solution"):
            problem = simple_problem(self.loglike, self._penalty)
            self.lasso_solution = problem.solve(**solve_args)
        elif lasso_solution is not None:
            self.lasso_solution = lasso_solution

        lasso_solution = self.lasso_solution # shorthand after setting it correctly above

        if not np.all(lasso_solution == 0):

            self.active = np.nonzero(lasso_solution != 0)[0]
            self.inactive = lasso_solution == 0
            self.active_signs = np.sign(lasso_solution[self.active])
            self._active_soln = lasso_solution[self.active]
            H = self.loglike.hessian(self.lasso_solution)
            H_AA = H[self.active][:,self.active]
            H_AAinv = np.linalg.inv(H_AA)
            Q = self.loglike.quadratic
            G_Q = Q.objective(self.lasso_solution, 'grad')
            G = self.loglike.gradient(self.lasso_solution) + G_Q
            G_A = G[self.active]
            G_I = self._G_I = G[self.inactive]
            dbeta_A = H_AAinv.dot(G_A)
            self.onestep_estimator = self._active_soln - dbeta_A
            self.active_penalized = self.feature_weights[self.active] != 0

            if self.active_penalized.sum():
                self._constraints = constraints(-np.diag(self.active_signs)[self.active_penalized],
                                                 (self.active_signs * dbeta_A)[self.active_penalized],
                                                 covariance=H_AAinv)
            else:
                self._constraints = constraints(np.identity(self.active.shape[0]),
                                                1.e12 * np.ones(self.active.shape[0]) * H_AAinv.max(), # XXX np.inf seems to fail tests
                                                covariance=H_AAinv)

            if self.inactive.sum():

                # inactive constraints

                H_IA = H[self.inactive][:,self.active]
                H_II = H[self.inactive][:,self.inactive]
                inactive_cov = H_II - H_IA.dot(H_AAinv).dot(H_IA.T)
                irrepresentable = H_IA.dot(H_AAinv)
                inactive_mean = irrepresentable.dot(-G_A)
                self._inactive_constraints = constraints(np.vstack([np.identity(self.inactive.sum()),
                                                                    -np.identity(self.inactive.sum())]),
                                                         np.hstack([self.feature_weights[self.inactive],
                                                                    self.feature_weights[self.inactive]]),
                                                         covariance=inactive_cov,
                                                         mean=inactive_mean)
                if not self._inactive_constraints(G_I):
                    warnings.warn('inactive constraint of KKT conditions not satisfied -- perhaps need to solve with more accuracy')

                if self.covariance_estimator is not None:

                    # make full constraints

                    # A: active
                    # I: inactive
                    # F: full, (A,I) stacked

                    _cov_FA = self.covariance_estimator(self.onestep_estimator,
                                                        self.active,
                                                        self.inactive)

                    _cov_IA = _cov_FA[len(self.active):]
                    _cov_AA = _cov_FA[:len(self.active)]

                    _beta_bar = self.onestep_estimator

                    if not self.ignore_inactive_constraints:
                        # X_{-E}^T(y - X_E \bar{\beta}_E)

                        _inactive_score = - G_I - inactive_mean

                        _indep_linear_part = _cov_IA.dot(np.linalg.inv(_cov_AA))

                        # we "fix" _nuisance, effectively conditioning on it

                        _nuisance = _inactive_score - _indep_linear_part.dot(_beta_bar)
                        _upper_lim = (self.feature_weights[self.inactive] - 
                                      _nuisance - 
                                      inactive_mean)
                        _lower_lim = (_nuisance + 
                                      self.feature_weights[self.inactive] +
                                      inactive_mean)

                        _upper_linear = _indep_linear_part
                        _lower_linear = -_indep_linear_part

                        C = self._constraints
                        _full_linear = np.vstack([C.linear_part,
                                                  _upper_linear,
                                                  _lower_linear])

                        _full_offset = np.hstack([C.offset,
                                                  _upper_lim,
                                                  _lower_lim])

                        self._constraints = constraints(_full_linear,
                                                        _full_offset,
                                                        covariance=_cov_AA)
                    else:
                        self._constraints.covariance[:] = _cov_AA

                    if not self._constraints(_beta_bar):
                        warnings.warn('constraints of KKT conditions on one-step estimator ' + 
                                      ' not satisfied -- perhaps need to solve with more' + 
                                      'accuracy')

            else:
                self._inactive_constraints = None
        else:
            self.active = []
            self.inactive = np.arange(lasso_solution.shape[0])
            self._constraints = None
            self._inactive_constraints = None
        return self.lasso_solution

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "lasso_solution"):
            self.fit()
        return self.lasso_solution

    @property
    def constraints(self):
        """
        Affine constraints for this LASSO problem.
        These are the constraints determined only
        by the active block.
        """
        return self._constraints

    @staticmethod
    def gaussian(X, 
                 Y, 
                 feature_weights, 
                 sigma=1., 
                 covariance_estimator=None,
                 quadratic=None):
        r"""
        Squared-error LASSO with feature weights.

        Objective function is 
        $$
        \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        $$

        where $\lambda$ is `feature_weights`.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        Y : ndarray
            Shape (n,) -- the response.

        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `feature_weights` to 0. If `feature_weights` is 
            a float, then all parameters are penalized equally.

        sigma : float (optional)
            Noise variance. Set to 1 if `covariance_estimator` is not None.
            This scales the loglikelihood by `sigma**(-2)`.

        covariance_estimator : callable (optional)
            If None, use the parameteric
            covariance estimate of the selected model.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic 
            coefficient to 0.

        Returns
        -------

        L : `selection.algorithms.lasso.lasso`
        
        Notes
        -----

        If not None, `covariance_estimator` should 
        take arguments (beta, active, inactive)
        and return an estimate of some of the
        rows and columns of the covariance of
        $(\bar{\beta}_E, \nabla \ell(\bar{\beta}_E)_{-E})$,
        the unpenalized estimator and the inactive
        coordinates of the gradient of the likelihood at
        the unpenalized estimator.

        """
        if covariance_estimator is not None:
            sigma = 1.
        loglike = glm.gaussian(X, Y, coef=1. / sigma**2, quadratic=quadratic)
        return lasso(loglike, np.asarray(feature_weights) / sigma**2,
                     covariance_estimator=covariance_estimator)

    @staticmethod
    def logistic(X, 
                 successes, 
                 feature_weights, 
                 trials=None, 
                 covariance_estimator=None,
                 quadratic=None):
        r"""
        Logistic LASSO with feature weights.

        Objective function is 
        $$
        \beta \mapsto \ell(X\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$

        where $\ell$ is the negative of the logistic 
        log-likelihood (half the logistic deviance)
        and $\lambda$ is `feature_weights`.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        successes : ndarray
            Shape (n,) -- response vector. An integer number of successes.
            For data that is proportions, multiply the proportions
            by the number of trials first.

        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `feature_weights` to 0. If `feature_weights` is 
            a float, then all parameters are penalized equally.

        trials : ndarray (optional)
            Number of trials per response, defaults to
            ones the same shape as Y. 

        covariance_estimator : optional
            If None, use the parameteric
            covariance estimate of the selected model.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic 
            coefficient to 0.

        Returns
        -------

        L : `selection.algorithms.lasso.lasso`
        
        Notes
        -----

        If not None, `covariance_estimator` should 
        take arguments (beta, active, inactive)
        and return an estimate of the covariance of
        $(\bar{\beta}_E, \nabla \ell(\bar{\beta}_E)_{-E})$,
        the unpenalized estimator and the inactive
        coordinates of the gradient of the likelihood at
        the unpenalized estimator.

        """
        loglike = glm.logistic(X, successes, trials=trials, quadratic=quadratic)
        return lasso(loglike, feature_weights,
                     covariance_estimator=covariance_estimator)

    @staticmethod
    def coxph(X, 
              times, 
              status, 
              feature_weights, 
              covariance_estimator=None,
              quadratic=None):
        r"""
        Cox proportional hazards LASSO with feature weights.

        Objective function is 
        $$
        \beta \mapsto \ell^{\text{Cox}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$

        where $\ell^{\text{Cox}}$ is the 
        negative of the log of the Cox partial
        likelihood and $\lambda$ is `feature_weights`.

        Uses Efron's tie breaking method.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        times : ndarray
            Shape (n,) -- the survival times.

        status : ndarray
            Shape (n,) -- the censoring status.

        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `feature_weights` to 0. If `feature_weights` is 
            a float, then all parameters are penalized equally.

        covariance_estimator : optional
            If None, use the parameteric
            covariance estimate of the selected model.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic 
            coefficient to 0.

        Returns
        -------

        L : `selection.algorithms.lasso.lasso`
        
        Notes
        -----

        If not None, `covariance_estimator` should 
        take arguments (beta, active, inactive)
        and return an estimate of the covariance of
        $(\bar{\beta}_E, \nabla \ell(\bar{\beta}_E)_{-E})$,
        the unpenalized estimator and the inactive
        coordinates of the gradient of the likelihood at
        the unpenalized estimator.

        """
        loglike = coxph_obj(X, times, status, quadratic=quadratic)
        return lasso(loglike, feature_weights,
                     covariance_estimator=covariance_estimator)

    @staticmethod
    def poisson(X, 
                counts, 
                feature_weights, 
                covariance_estimator=None,
                quadratic=None):
        r"""
        Poisson log-linear LASSO with feature weights.

        Objective function is 
        $$
        \beta \mapsto \ell^{\text{Poisson}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$

        where $\ell^{\text{Poisson}}$ is the negative
        of the log of the Poisson likelihood (half the deviance)
        and $\lambda$ is `feature_weights`.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        counts : ndarray
            Shape (n,) -- the response.

        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `feature_weights` to 0. If `feature_weights` is 
            a float, then all parameters are penalized equally.

        covariance_estimator : optional
            If None, use the parameteric
            covariance estimate of the selected model.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic 
            coefficient to 0.

        Returns
        -------

        L : `selection.algorithms.lasso.lasso`
        
        Notes
        -----

        If not None, `covariance_estimator` should 
        take arguments (beta, active, inactive)
        and return an estimate of the covariance of
        $(\bar{\beta}_E, \nabla \ell(\bar{\beta}_E)_{-E})$,
        the unpenalized estimator and the inactive
        coordinates of the gradient of the likelihood at
        the unpenalized estimator.

        """
        loglike = glm.poisson(X, counts, quadratic=quadratic)
        return lasso(loglike, feature_weights,
                     covariance_estimator=covariance_estimator)

    @staticmethod
    def sqrt_lasso(X, 
                   Y, 
                   feature_weights, 
                   quadratic=None,
                   covariance='parametric',
                   sigma_estimate='truncated',
                   solve_args={'min_its':200}):
        r"""
        Use sqrt-LASSO to choose variables.

        Objective function is 
        $$
        \beta \mapsto \|Y-X\beta\|_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        $$

        where $\lambda$ is `feature_weights`. After solving the problem
        treat as if `gaussian` with implied variance and choice of 
        multiplier. See arxiv.org/abs/1504.08031 for details.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        Y : ndarray
            Shape (n,) -- the response.

        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `feature_weights` to 0. If `feature_weights` is 
            a float, then all parameters are penalized equally.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic 
            coefficient to 0.

        covariance : str
            One of 'parametric' or 'sandwich'. Method
            used to estimate covariance for inference
            in second stage.

        sigma_estimate : str
            One of 'truncated' or 'OLS'. Method
            used to estimate $\sigma$ when using
            parametric covariance.

        solve_args : dict
            Arguments passed to solver.

        Returns
        -------

        L : `selection.algorithms.lasso.lasso`
        
        Notes
        -----

        Unlike other variants of LASSO, this
        solves the problem on construction as the active
        set is needed to find equivalent gaussian LASSO.

        Assumes parametric model is correct for inference,
        i.e. does not accept a covariance estimator.

        """

        n, p = X.shape

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(p) * feature_weights
        feature_weights = np.asarray(feature_weights)

        # TODO: refits sqrt lasso more than once -- make an override for avoiding refitting?

        soln = solve_sqrt_lasso(X, Y, weights=feature_weights, quadratic=quadratic, solve_args=solve_args)[0]

        # find active set, and estimate of sigma

        active = (soln != 0)
        nactive = active.sum()

        if nactive:

            subgrad = np.sign(soln[active]) * feature_weights[active]
            X_E = X[:,active]
            X_Ei = np.linalg.pinv(X_E)
            sigma_E = np.linalg.norm(Y - X_E.dot(X_Ei.dot(Y))) / np.sqrt(n - nactive)
            multiplier = np.sqrt((n - nactive) / (1 - np.linalg.norm(X_Ei.T.dot(subgrad))**2))

            # check truncation interval for sigma_E

            # the KKT conditions imply an inequality like
            # \hat{\sigma}_E \cdot LHS \leq RHS

            penalized = feature_weights[active] != 0

            if penalized.sum():
                D_E = np.sign(soln[active][penalized]) # diagonal matrix of signs
                LHS = D_E * np.linalg.solve(X_E.T.dot(X_E), subgrad)[penalized]
                RHS = D_E * X_Ei.dot(Y)[penalized] 

                ratio = RHS / LHS

                group1 = LHS > 0
                upper_bound = np.inf
                if group1.sum():
                    upper_bound = min(upper_bound, np.min(ratio[group1])) # necessarily these will have RHS > 0

                group2 = (LHS <= 0) * (RHS <= 0) # we can ignore the other possibility since this gives a lower bound of 0
                lower_bound = 0
                if group2.sum():
                    lower_bound = max(lower_bound, np.max(ratio[group2]))

                upper_bound /= multiplier
                lower_bound /= multiplier

            else:
                lower_bound = 0
                upper_bound = np.inf

            _sigma_estimator_args = (sigma_E, 
                                     n - nactive,
                                     lower_bound, 
                                     upper_bound)

            if sigma_estimate == 'truncated':
                _sigma_hat = estimate_sigma(*_sigma_estimator_args)
            elif sigma_estimate == 'OLS':
                _sigma_hat = sigma_E
            else:
                raise ValueError('sigma_estimate must be one of ["truncated", "OLS"]')
        else:
            _sigma_hat = np.linalg.norm(Y) / np.sqrt(n)
            multiplier = np.sqrt(n)
            sigma_E = _sigma_hat

        # XXX how should quadratic be changed?
        # multiply everything by sigma_E?

        if quadratic is not None:
            qc = quadratic.collapsed()
            qc.coef *= np.sqrt(n - nactive) / sigma_E
            qc.linear_term *= np.sqrt(n - nactive) / sigma_E
            quadratic = qc

        loglike = glm.gaussian(X, Y, quadratic=quadratic)

        if covariance == 'parametric':
            cov_est = glm_parametric_estimator(loglike, dispersion=_sigma_hat)
        elif covariance == 'sandwich':
            cov_est = glm_sandwich_estimator(loglike, B=2000)
        else:
            raise ValueError('covariance must be one of ["parametric", "sandwich"]')

        L = lasso(loglike, feature_weights * multiplier * sigma_E,
                  covariance_estimator=cov_est,
                  ignore_inactive_constraints=True)

        # these arguments are reused for data carving

        if nactive:
            L._sigma_hat = _sigma_hat
            L._sigma_estimator_args = _sigma_estimator_args
            L._weight_multiplier = multiplier * sigma_E
            L._multiplier = multiplier
            L.lasso_solution = soln

        return L

    def summary(self, alternative='twosided', alpha=0.05, UMAU=False,
                compute_intervals=False):
        """
        Summary table for inference adjusted for selection.

        Parameters
        ----------

        alternative : str
            One of ["twosided","onesided"]

        Returns
        -------

        pval_summary : np.recarray
            Array with one entry per active variable.
            Columns are 'variable', 'pval', 'lasso', 'onestep', 'lower_trunc', 'upper_trunc', 'sd'.

        alpha : float
            Form (1-alpha)*100% selective confidence intervals.

        UMAU : bool
            If True, form the UMAU intervals (slow, perhaps less stable).

        compute_intervals : bool
            Should we compute confidence intervals?

        """

        if alternative not in ['twosided', 'onesided']:
            raise ValueError("alternative must be one of ['twosided', 'onesided']")

        result = []
        C = self.constraints
        if C is not None:
            one_step = self.onestep_estimator
            for i in range(one_step.shape[0]):
                eta = np.zeros_like(one_step)
                eta[i] = self.active_signs[i]
                _alt = {"onesided":'greater',
                        'twosided':"twosided"}[alternative]
                if C.linear_part.shape[0] > 0: # there were some constraints
                    _pval = C.pivot(eta, one_step, alternative=_alt)
                else:
                    obs = (eta * one_step).sum()
                    sd = np.sqrt((eta * C.covariance.dot(eta)))
                    Z = obs / sd
                    _pval = 2 * ndist.sf(np.fabs(Z))

                if compute_intervals:
                    if C.linear_part.shape[0] > 0: # there were some constraints
                        _interval = C.interval(eta, one_step,
                                               alpha=alpha,
                                               UMAU=UMAU)
                        _interval = sorted([_interval[0] * self.active_signs[i],
                                            _interval[1] * self.active_signs[i]])
                    else:
                        _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                                     obs + ndist.ppf(1 - alpha / 2) * sd)
                else:
                    _interval = [np.nan, np.nan]
                _bounds = np.array(C.bounds(eta, one_step))
                sd = _bounds[-1]
                lower_trunc, est, upper_trunc = sorted(_bounds[:3] * self.active_signs[i])

                result.append((self.active[i],
                               _pval,
                               self.lasso_solution[self.active[i]],
                               one_step[i],
                               _interval[0],
                               _interval[1],
                               lower_trunc,
                               upper_trunc,
                               sd))
                
        df = pd.DataFrame(index=self.active,
                          data=dict([(n, d) for n, d in zip(['variable',
                                                             'pval', 
                                                             'lasso', 
                                                             'onestep', 
                                                             'lower_confidence', 
                                                             'upper_confidence',
                                                             'lower_trunc',
                                                             'upper_trunc',
                                                             'sd'], 
                                                            np.array(result).T)]))
        return df


def nominal_intervals(lasso_obj):
    """
    Intervals for OLS parameters of active variables
    that have not been adjusted for selection.
    """
    unadjusted_intervals = []

    if lasso_obj.active is not []:
        SigmaE = lasso_obj.constraints.covariance
        for i in range(lasso_obj.active.shape[0]):
            eta = np.ones_like(lasso_obj.onestep_estimator)
            eta[i] = 1.
            center = lasso_obj.onestep_estimator[i]
            width = ndist.ppf(1-lasso_obj.alpha/2.) * np.sqrt(SigmaE[i,i])
            _interval = [center-width, center+width]
            unadjusted_intervals.append((lasso_obj.active[i], eta, center,
                                         _interval))
    return unadjusted_intervals

def glm_sandwich_estimator(loss, B=1000):
    """
    Bootstrap estimator of covariance of 
    
    .. math::
    
        (\bar{\beta}_E, X_{-E}^T(y-X_E\bar{\beta}_E)

    the OLS estimator of population regression 
    coefficients and inactive correlation with the
    OLS residuals.

    Returns
    -------

    estimator : callable
        Takes arguments (beta, active, inactive)

    """
    
    def _estimator(loss, B, beta, active, inactive):
        
        X, Y = loss.data
        n, p = X.shape # shorthand

        beta_full = np.zeros(p)
        beta_full[active] = beta

        # make sure active / inactive are bool

        active_bool = np.zeros(p, np.bool)
        active_bool[active] = 1

        inactive_bool = np.zeros(p, np.bool)
        inactive_bool[inactive] = 1

        bootstrapper = pairs_bootstrap_glm(loss, 
                                           active_bool, 
                                           beta_full=beta_full,
                                           inactive=inactive_bool)[0]

        nactive = active_bool.sum()
        first_moment = np.zeros(p)
        second_moment = np.zeros((p, nactive))

        for b in range(B):
            indices = np.random.choice(n, n, replace=True)
            Z_star = bootstrapper(indices)
            first_moment += Z_star
            second_moment += np.multiply.outer(Z_star, Z_star[:nactive])

        first_moment /= B
        second_moment /= B

        cov = second_moment - np.multiply.outer(first_moment, 
                                                first_moment[:nactive])

        return cov

    return functools.partial(_estimator, loss, B)

def glm_parametric_estimator(loglike, dispersion=None):
    """
    Parametric estimator of covariance of 
    
    .. math::
    
        (\bar{\beta}_E, X_{-E}^T(y-\nabla \ell(X_E\bar{\beta}_E))

    the OLS estimator of population regression 
    coefficients and inactive correlation with the
    OLS residuals.

    If `sigma` is None, it computes usual unbiased estimate 
    of variance in Gaussian model and plugs it in, 
    assuming parametric form is correct.

    Returns
    -------

    estimator : callable
        Takes arguments (beta, active, inactive)

    """
    
    def _estimator(loglike, dispersion, beta, active, inactive):
        
        X, Y = loglike.data
        n, p = X.shape
        nactive = len(active)

        linear_predictor = X[:,active].dot(beta)
        W = loglike.saturated_loss.hessian(linear_predictor)
        Sigma_A = X[:,active].T.dot(W[:, None] * X[:,active]) 
        Sigma_Ainv = np.linalg.inv(Sigma_A)

        _unscaled = np.zeros((p, len(active)))
        _unscaled[:nactive] = Sigma_Ainv

        # the lower block is left at 0 because
        # under the parametric model, these pieces
        # are independent

        # if no dispersion, use Pearson's X^2

        if dispersion is None:
            eta = X[:,active].dot(beta)
            dispersion= ((loglike.saturated_loss.smooth_objective(eta, 'grad'))**2 / W).sum() / (n - nactive)

        _unscaled *= dispersion**2

        return _unscaled

    return functools.partial(_estimator, loglike, dispersion)

def standard_lasso(X, y, sigma=1, lam_frac=1., **solve_args):
    """
    Fit a LASSO with a default choice of Lagrange parameter
    equal to `lam_frac` times $\sigma \cdot E(|X^T\epsilon|)$
    with $\epsilon$ IID N(0,1).

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    lam_frac : float
        Multiplier for choice of $\lambda$

    solve_args : keyword args
        Passed to `regreg.problems.simple_problem.solve`.

    Returns
    -------

    lasso_selection : `lasso`
         Instance of `lasso` after fitting. 

    """
    n, p = X.shape
    lam = lam_frac * np.mean(np.fabs(X.T.dot(np.random.standard_normal((n, 50000)))).max(0)) * sigma
    lasso_selector = lasso.gaussian(X, y, lam, sigma=sigma)
    lasso_selector.fit(**solve_args)

    return lasso_selector

class data_carving(lasso):

    """

    Notes
    -----

    Even if a covariance estimator is supplied,
    we assume that we can drop inactive constraints, 
    i.e. the same (asymptotic) independence that
    holds for parametric model is assumed to hold here
    as well.

    """

    def __init__(self, 
                 loglike_select,
                 loglike_inference,
                 loglike_full,
                 feature_weights,
                 covariance_estimator=None):

        lasso.__init__(self, loglike_select, feature_weights, covariance_estimator=covariance_estimator)
        self.loglike_inference = loglike_inference
        self.loglike_full = loglike_full

    @classmethod
    def gaussian(klass,
                 X, 
                 Y, 
                 feature_weights, 
                 split_frac=0.9,
                 sigma=1.,
                 stage_one=None):
        
        n, p = X.shape
        if stage_one is None:
            splitn = int(n*split_frac)
            indices = np.arange(n)
            np.random.shuffle(indices)
            stage_one = indices[:splitn]
            stage_two = indices[splitn:]
        else:
            stage_two = [i for i in np.arange(n) if i not in stage_one]
        Y1, X1 = Y[stage_one], X[stage_one]
        Y2, X2 = Y[stage_two], X[stage_two]

        loglike = glm.gaussian(X, Y, coef=1. / sigma**2)
        loglike1 = glm.gaussian(X1, Y1, coef=1. / sigma**2)
        loglike2 = glm.gaussian(X2, Y2, coef=1. / sigma**2)

        return klass(loglike1, loglike2, loglike, feature_weights / sigma**2)

    @classmethod
    def logistic(klass,
                 X, 
                 successes,
                 feature_weights, 
                 trials=None,
                 split_frac=0.9,
                 sigma=1.,
                 stage_one=None):
        
        n, p = X.shape
        if stage_one is None:
            splitn = int(n*split_frac)
            indices = np.arange(n)
            np.random.shuffle(indices)
            stage_one = indices[:splitn]
            stage_two = indices[splitn:]
        else:
            stage_two = [i for i in np.arange(n) if i not in stage_one]

        if trials is None:
            trials = np.ones_like(successes)

        successes1, X1, trials1 = successes[stage_one], X[stage_one], trials[stage_one]
        successes2, X2, trials2 = successes[stage_two], X[stage_two], trials[stage_two]

        loglike = glm.logistic(X, successes, trials=trials)
        loglike1 = glm.logistic(X1, successes1, trials=trials1)
        loglike2 = glm.logistic(X2, successes2, trials=trials2)

        return klass(loglike1, loglike2, loglike, feature_weights)

    @classmethod
    def poisson(klass,
                X, 
                counts,
                feature_weights, 
                split_frac=0.9,
                sigma=1.,
                stage_one=None):
        
        n, p = X.shape
        if stage_one is None:
            splitn = int(n*split_frac)
            indices = np.arange(n)
            np.random.shuffle(indices)
            stage_one = indices[:splitn]
            stage_two = indices[splitn:]
        else:
            stage_two = [i for i in np.arange(n) if i not in stage_one]

        counts1, X1 = counts[stage_one], X[stage_one]
        counts2, X2 = counts[stage_two], X[stage_two]

        loglike = glm.poisson(X, counts)
        loglike1 = glm.poisson(X1, counts1)
        loglike2 = glm.poisson(X2, counts2)

        return klass(loglike1, loglike2, loglike, feature_weights)

    @classmethod
    def coxph(klass,
              X, 
              times, 
              status, 
              feature_weights, 
              split_frac=0.9,
              sigma=1.,
              stage_one=None):
        
        n, p = X.shape
        if stage_one is None:
            splitn = int(n*split_frac)
            indices = np.arange(n)
            np.random.shuffle(indices)
            stage_one = indices[:splitn]
            stage_two = indices[splitn:]
        else:
            stage_two = [i for i in np.arange(n) if i not in stage_one]

        times1, X1, status1 = times[stage_one], X[stage_one], status[stage_one]
        times2, X2, status2 = times[stage_two], X[stage_two], status[stage_two]

        loglike = coxph_obj(X, times, status)
        loglike1 = coxph_obj(X1, times1, status1)
        loglike2 = coxph_obj(X2, times2, status2)

        return klass(loglike1, loglike2, loglike, feature_weights)

    @classmethod
    def sqrt_lasso(klass,
                   X, 
                   Y, 
                   feature_weights, 
                   split_frac=0.9,
                   stage_one=None,
                   solve_args={'min_its':200}):
        
        n, p = X.shape

        if stage_one is None:
            splitn = int(n*split_frac)
            indices = np.arange(n)
            np.random.shuffle(indices)
            stage_one = indices[:splitn]
            stage_two = indices[splitn:]
        else:
            stage_two = [i for i in np.arange(n) if i not in stage_one]

        Y1, X1 = Y[stage_one], X[stage_one]
        Y2, X2 = Y[stage_two], X[stage_two]

        # TODO: refits sqrt lasso more than once

        L = lasso.sqrt_lasso(X1, Y1, feature_weights, solve_args=solve_args)
        soln = L.fit(solve_args=solve_args)
        _sigma_E1, _df1, _lower, _upper = L._sigma_estimator_args
        _df2 = max(len(stage_two) - len(L.active), 0)
        if _df2:
            X_E2 = X2[:,L.active]
            _sigma_E2 = np.linalg.norm(Y2 - X_E2.dot(np.linalg.pinv(X_E2).dot(Y2))) / (len(stage_two) - len(L.active))
            _sigma_hat = estimate_sigma(np.sqrt((_sigma_E1**2 * _df1 + _sigma_E2**2 * _df2) / (_df1 + _df2)),
                                        _df1,
                                        _lower,
                                        _upper,
                                        untruncated_df=_df2)
        else:
            _sigma_hat = L._sigma_hat

        cov_est = glm_parametric_estimator(L.loglike, dispersion=_sigma_hat)

        loglike = glm.gaussian(X, Y)
        loglike1 = glm.gaussian(X1, Y1)
        loglike2 = glm.gaussian(X2, Y2)

        L = klass(loglike1, loglike2, loglike, feature_weights * L._weight_multiplier,
                  covariance_estimator=cov_est)
        L.lasso_solution = soln
        return L

    def fit(self, solve_args={'tol':1.e-12, 'min_its':50}):

        lasso.fit(self, solve_args=solve_args)

        n1 = self.loglike.get_data()[0].shape[0]
        n = self.loglike_full.get_data()[0].shape[0]

        _feature_weights = self.feature_weights.copy()
        _feature_weights[self.active] = 0.
        _feature_weights[self.inactive] = np.inf
        
        _unpenalized_problem = simple_problem(self.loglike_full, 
                                              weighted_l1norm(_feature_weights, lagrange=1.))
        _unpenalized = _unpenalized_problem.solve(**solve_args)
        _unpenalized_active = _unpenalized[self.active]

        s = len(self.active)

        if self.covariance_estimator is None:
            H = self.loglike_full.hessian(_unpenalized)
            H_AA = H[self.active][:,self.active]
            _cov_block = np.linalg.inv(H_AA)
            self._carve_invcov = H_AA
        else:
            C = self.covariance_estimator(_unpenalized_active, self.active, self.inactive)
            _cov_block = C[:len(self.active)][:,:len(self.active)]
            self._carve_invcov = np.linalg.pinv(_cov_block)

        _subsample_block = (n * 1. / n1) * _cov_block
        _carve_cov = np.zeros((2*s,2*s))
        _carve_cov[:s][:,:s] = _cov_block
        _carve_cov[s:][:,:s] = _subsample_block
        _carve_cov[:s][:,s:] = _subsample_block
        _carve_cov[s:][:,s:] = _subsample_block

        _carve_linear_part = self._constraints.linear_part.dot(np.identity(2*s)[s:])
        _carve_offset = self._constraints.offset
        self._carve_constraints = constraints(_carve_linear_part,
                                              _carve_offset,
                                              covariance=_carve_cov)

        self._carve_feasible = np.hstack([_unpenalized_active, self.onestep_estimator])
        self._unpenalized_active = _unpenalized_active

    def hypothesis_test(self,
                        variable,
                        burnin=2000,
                        ndraw=8000,
                        compute_intervals=False):

        if variable not in self.active:
            raise ValueError('expecting an active variable')

        # shorthand
        j = list(self.active).index(variable) 
        twice_s = self._carve_constraints.linear_part.shape[1] 
        s = sparsity = int(twice_s / 2)

        keep = np.ones(s, np.bool)
        keep[j] = 0
        conditioning = self._carve_invcov.dot(np.identity(twice_s)[:s])[keep]

        contrast = np.zeros(2*s)
        contrast[j] = 1.

        # condition to remove dependence on nuisance parameters
        if len(self.active) > 1: 
            conditional_law = self._carve_constraints.conditional(conditioning,
                                                                  conditioning.dot(self._carve_feasible))
        else:
            conditional_law = self._carve_constraints

        observed = (contrast * self._carve_feasible).sum()

        if self._carve_constraints.linear_part.shape[0] > 0:

            _, _, _, family = gibbs_test(conditional_law,
                                         self._carve_feasible,
                                         contrast,
                                         sigma_known=True,
                                         white=False,
                                         ndraw=ndraw,
                                         burnin=burnin,
                                         how_often=10,
                                         UMPU=False)

            pval = family.cdf(0, observed)
            pval = 2 * min(pval, 1 - pval)
        
        else: # only unpenalized coefficients were nonzero, no constraints

            sd = np.sqrt((contrast * self._carve_constraints.covariance.dot(contrast)).sum())
            Z = observed / sd
            pval = 2 * ndist.sf(np.fabs(Z))

        return pval

class data_splitting(data_carving):

    def fit(self, solve_args={'tol':1.e-12, 'min_its':500}, use_full_cov=True):

        lasso.fit(self, solve_args=solve_args)

        _feature_weights = self.feature_weights.copy()
        _feature_weights[self.active] = 0.
        _feature_weights[self.inactive] = np.inf
        
        _unpenalized_problem = simple_problem(self.loglike_inference,
                                              weighted_l1norm(_feature_weights, lagrange=1.))
        _unpenalized = _unpenalized_problem.solve(**solve_args)

        self._unpenalized_active = _unpenalized[self.active]

        if use_full_cov:
            H = self.loglike_full.hessian(_unpenalized)
            n_inference = self.loglike_inference.data[0].shape[0]
            n_full = self.loglike_full.data[0].shape[0]
            H *= (1. * n_inference / n_full)
        else:
            H = self.loglike_inference.hessian(_unpenalized)

        H_AA = H[self.active][:,self.active]
        self._cov_inference = np.linalg.inv(H_AA)

    def hypothesis_test(self,
                        variable,
                        df=np.inf):
        """

        Wald test for an active variable.

        """
        if variable not in self.active:
            raise ValueError('expecting an active variable')

        # shorthand
        j = list(self.active).index(variable) 

        Z = self._unpenalized_active[j] / np.sqrt(self._cov_inference[j,j])

        if df == np.inf:
            return 2 * ndist.sf(np.abs(Z))
        else:
            return 2 * tdist.sf(np.abs(Z), df)

def _data_carving_deprec(X, y, 
                        lam_frac=2.,
                        sigma=1., 
                        stage_one=None,
                        split_frac=0.9,
                        coverage=0.95, 
                        ndraw=8000,
                        burnin=2000,
                        splitting=False,
                        compute_intervals=True,
                        UMPU=False):

    """
    Fit a LASSO with a default choice of Lagrange parameter
    equal to `lam_frac` times $\sigma \cdot E(|X^T\epsilon|)$
    with $\epsilon$ IID N(0,1) on a proportion (`split_frac`) of
    the data.

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$. Defaults to 2.

    coverage : float
        Coverage for selective intervals. Defaults to 0.95.

    stage_one : [np.array(np.int), None] (optional)
        Index of data points to be used in  first stage.
        If None, a randomly chosen set of entries is used based on
        `split_frac`.

    split_frac : float (optional)
        What proportion of the data to use in the first stage?
        Defaults to 0.9.

    ndraw : int (optional)
        How many draws to keep from Gibbs hit-and-run sampler.
        Defaults to 8000.

    burnin : int (optional)
        Defaults to 2000.

    splitting : bool (optional)
        If True, also return splitting pvalues and intervals.

    compute_intervals : bool (optional)
        Compute selective intervals?

    UMPU : bool (optional)
        Perform the UMPU test?
      
    Returns
    -------

    results : [(variable, pvalue, interval)
        Indices of active variables, 
        selected (twosided) pvalue and selective interval.
        If splitting, then each entry also includes
        a (split_pvalue, split_interval) using stage_two
        for inference.

    stage_one : `lasso`
        Results of fitting LASSO to stage one data.

    """

    n, p = X.shape
    first_stage, stage_one, stage_two = split_model(X, y,
                                                    sigma=sigma,
                                                    lam_frac=lam_frac,
                                                    split_frac=split_frac,
                                                    stage_one=stage_one)
    splitn = stage_one.shape[0]

    L = first_stage # shorthand
    s = sparsity = L.active.shape[0]

    if splitn < n:

        # JT: this is all about computing constraints for active
        # variables -- we already have this!

        # quantities related to models fit on
        # stage_one and full dataset

        y1, X1 = y[stage_one], X[stage_one]
        X_E = X[:,L.active] 
        X_Ei = np.linalg.pinv(X_E)
        X_E1 = X1[:,L.active]
        X_Ei1 = np.linalg.pinv(X_E1)

        inv_info_E = X_Ei.dot(X_Ei.T)
        inv_info_E1 = X_Ei1.dot(X_Ei1.T)

        beta_E = X_Ei.dot(y)
        beta_E1 = X_Ei1.dot(y[stage_one])

        if n - splitn > s:

            linear_part = np.zeros((s, 2*s))
            linear_part[:, s:] = -np.diag(L.active_signs)
            b = L.constraints.offset
            con = constraints(linear_part, b)

            # specify covariance of 2s Gaussian vector

            cov = np.zeros((2*s, 2*s))
            cov[:s, :s] = inv_info_E
            cov[s:, :s] = inv_info_E
            cov[:s, s:] = inv_info_E
            cov[s:, s:] = inv_info_E1

            con.covariance[:] = cov * sigma**2

            # for the conditional law
            # we will change the linear function for each coefficient

            selector = np.zeros((s, 2*s))
            selector[:, :s]  = np.identity(s)
            conditional_linear = X_E.T.dot(X_E).dot(selector) 

            # a valid initial condition

            initial = np.hstack([beta_E, beta_E1]) 
            OLS_func = selector

        else:

            linear_part = np.zeros((s, s + n - splitn))
            linear_part[:, :s] = -np.diag(L.active_signs)
            b = L.constraints.offset
            con = constraints(linear_part, b)

            # specify covariance of Gaussian vector

            cov = np.zeros((s + n - splitn, s + n - splitn))
            cov[:s, :s] = inv_info_E1
            cov[s:, :s] = 0
            cov[:s, s:] = 0
            cov[s:, s:] = np.identity(n - splitn) 

            con.covariance[:] = cov * sigma**2

            conditional_linear = np.zeros((s, s + n - splitn))
            conditional_linear[:, :s]  = np.linalg.pinv(inv_info_E1)
            conditional_linear[:, s:] = X[stage_two,:][:,L.active].T

            selector1 = np.zeros((s, s + n - splitn))
            selector1[:, :s]  = np.identity(s)
            selector2 = np.zeros((n - splitn, s + n - splitn))
            selector2[:, s:]  = np.identity(n - splitn)

            # write the OLS estimates of full model in terms of X_E1^{dagger}y_1, y2

            OLS_func = inv_info_E.dot(conditional_linear) 

            # a valid initial condition

            initial = np.hstack([beta_E1, y[stage_two]]) 
            
        pvalues = []
        intervals = []

        if splitting:
            y2, X2 = y[stage_two], X[stage_two]
            X_E2 = X2[:,L.active]
            X_Ei2 = np.linalg.pinv(X_E2)
            beta_E2 = X_Ei2.dot(y2)
            inv_info_E2 = X_Ei2.dot(X_Ei2.T)

            splitting_pvalues = []
            splitting_intervals = []

            if n - splitn < s:
                warnings.warn('not enough data for second stage of sample splitting')

            split_cutoff = np.fabs(ndist.ppf((1. - coverage) / 2))

        # compute p-values intervals

        cov_inv = np.linalg.pinv(con.covariance)

        for j in range(X_E.shape[1]):

            keep = np.ones(s, np.bool)
            keep[j] = 0

            eta = OLS_func[j]

            con_cp = copy(con)
            conditional_law = con_cp.conditional(conditional_linear[keep], \
                                                 X_E.T.dot(y)[keep])
            
            # tilt so that samples are closer to observed values
            # the multiplier should be the pseudoMLE so that
            # the observed value is likely 

            observed = (initial * eta).sum()

            if compute_intervals:
                _, _, _, family = gibbs_test(conditional_law,
                                             initial, 
                                             eta,
                                             sigma_known=True,
                                             white=False,
                                             ndraw=ndraw,
                                             burnin=burnin,
                                             how_often=10,
                                             UMPU=UMPU,
                                             tilt=conditional_law.covariance.dot(eta))

                lower_lim, upper_lim = family.equal_tailed_interval(observed, 1 - coverage)

                # in the model we've chosen, the parameter beta is associated
                # to the natural parameter as below
                # exercise: justify this!

                lower_lim_final = eta.dot(conditional_law.covariance.dot(eta)) * lower_lim
                upper_lim_final = eta.dot(conditional_law.covariance.dot(eta)) * upper_lim

                intervals.append((lower_lim_final, upper_lim_final))
            else: # we do not really need to tilt just for p-values
                _, _, _, family = gibbs_test(conditional_law,
                                             initial, 
                                             eta,
                                             sigma_known=True,
                                             white=False,
                                             ndraw=ndraw,
                                             burnin=burnin,
                                             how_often=10,
                                             UMPU=UMPU)
                intervals.append((np.nan, np.nan))

            pval = family.cdf(0, observed)
            pval = 2 * min(pval, 1 - pval)

            pvalues.append(pval)

            if splitting:

                if s < n - splitn: # enough data to generically
                                   # test hypotheses. proceed as usual

                    split_pval = ndist.cdf(beta_E2[j] / (np.sqrt(inv_info_E2[j,j]) * sigma))
                    split_pval = 2 * min(split_pval, 1. - split_pval)
                    splitting_pvalues.append(split_pval)

                    splitting_interval = (beta_E2[j] - 
                                          split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma,
                                          beta_E2[j] + 
                                          split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma)
                    splitting_intervals.append(splitting_interval)
                else:
                    splitting_pvalues.append(np.random.sample())
                    splitting_intervals.append((np.nan, np.nan))

        if not splitting:
            return zip(L.active, 
                       pvalues,
                       intervals), L
        else:
            return zip(L.active, 
                       pvalues,
                       intervals,
                       splitting_pvalues,
                       splitting_intervals), L
    else:
        pvalues = [p for _, p in L.summary("twosided")['pval']]
        intervals = np.array([L.intervals['lower'], L.intervals['upper']]).T
        if splitting:
            splitting_pvalues = np.random.sample(len(pvalues))
            splitting_intervals = [(np.nan, np.nan) for _ in 
                                   range(len(pvalues))]

            return zip(L.active, 
                       pvalues, 
                       intervals,
                       splitting_pvalues,
                       splitting_intervals), L
        else:
            return zip(L.active, 
                       pvalues,
                       intervals), L
            
def split_model(X, y, 
                sigma=1, 
                lam_frac=1.,
                split_frac=0.9,
                stage_one=None):

    """
    Fit a LASSO with a default choice of Lagrange parameter
    equal to `lam_frac` times $\sigma \cdot E(|X^T\epsilon|)$
    with $\epsilon$ IID N(0,1) on a proportion (`split_frac`) of
    the data.

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$. Defaults to 2.

    split_frac : float (optional)
        What proportion of the data to use in the first stage?
        Defaults to 0.9.

    stage_one : [np.array(np.int), None] (optional)
        Index of data points to be used in  first stage.
        If None, a randomly chosen set of entries is used based on
        `split_frac`.

    Returns
    -------

    first_stage : `lasso`
        Lasso object from stage one.

    stage_one : np.array(int)
        Indices used for stage one.

    stage_two : np.array(int)
        Indices used for stage two.

    """

    n, p = X.shape
    if stage_one is None:
        splitn = int(n*split_frac)
        indices = np.arange(n)
        np.random.shuffle(indices)
        stage_one = indices[:splitn]
        stage_two = indices[splitn:]
    else:
        stage_two = [i for i in np.arange(n) if i not in stage_one]
    y1, X1 = y[stage_one], X[stage_one]

    first_stage = standard_lasso(X1, y1, sigma=sigma, lam_frac=lam_frac)
    return first_stage, stage_one, stage_two

def additive_noise(X, 
                   y, 
                   sigma, 
                   lam_frac=1.,
                   perturb_frac=0.2, 
                   y_star=None,
                   coverage=0.95,
                   ndraw=8000, 
                   compute_intervals=True,
                   burnin=2000):


    """
    
    Additive noise LASSO.

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$. Defaults to 2.

    perturb_frac : float (optional)
        How much noise to add? Noise added has variance
        proportional to existing variance.

    coverage : float
        Coverage for selective intervals. Defaults to 0.95.

    ndraw : int (optional)
        How many draws to keep from Gibbs hit-and-run sampler.
        Defaults to 8000.

    burnin : int (optional)
        Defaults to 2000.

    compute_intervals : bool (optional)
        Compute selective intervals?
      
    Returns
    -------

    results : [(variable, pvalue, interval)
        Indices of active variables, 
        selected (twosided) pvalue and selective interval.
        If splitting, then each entry also includes
        a (split_pvalue, split_interval) using stage_two
        for inference.

    randomized_lasso : `lasso`
        Results of fitting LASSO to randomized data.

    """

    n, p = X.shape

    # Add some noise to y and fit the LASSO at a fixed lambda

    gamma = np.sqrt(perturb_frac) * sigma 
    sigma_star = np.sqrt(sigma**2 + gamma**2)
    lam = lam_frac * np.mean(np.fabs(X.T.dot(np.random.standard_normal((n, 5000)))).max(0)) * sigma_star
    y_star = y + np.random.standard_normal(n) * gamma

    randomized_lasso = L = standard_lasso(X, y_star, sigma=sigma_star, lam_frac=lam_frac)
    L.fit()

    # Form the constraint matrix on (y,y^*)
    X_E = X[:,L.active]
    X_Ei = np.linalg.pinv(X_E)
    Cov_E = X_Ei.dot(X_Ei.T)
    W_E = Cov_E.dot(L.active_signs)

    pvalues = []
    intervals = []

    beta_E = X_Ei.dot(y)

    # compute each pvalue
    for j in range(X_E.shape[1]):
        s_obs = L.active.shape[0]
        keep = np.ones(s_obs, np.bool)
        keep[j] = 0

        # form the 2s Gaussian vector we will condition on

        X_minus_j = X_E[:,keep]
        P_minus_j = X_minus_j.dot(np.linalg.pinv(X_minus_j))
        R_minus_j = np.identity(n) - P_minus_j

        theta_E = L.active_signs * (X_Ei.dot(P_minus_j.dot(y)) - lam * W_E)
        scale = np.sqrt(Cov_E[j,j])
        kappa = 1. / scale**2
        alpha_E = kappa * L.active_signs * Cov_E[j]
        A = np.hstack([-alpha_E.reshape((s_obs,1)), np.identity(s_obs)])
        con = constraints(A, theta_E)
        cov = np.zeros((s_obs+1, s_obs+1))
        cov[0,0] = scale**2 * sigma**2
        cov[1:,1:] = Cov_E * gamma**2 * np.outer(L.active_signs, L.active_signs)
        con.covariance[:] = cov
        initial = np.zeros(s_obs+1)
        initial[0] = beta_E[j]
        initial[1:] = -X_Ei.dot(y_star-y) * L.active_signs
        eta = np.zeros(s_obs+1)
        eta[0] = 1.

        observed = (initial * eta).sum()

        if compute_intervals:
            _, _, _, family = gibbs_test(con,
                                         initial,
                                         eta,
                                         UMPU=False,
                                         sigma_known=True,
                                         ndraw=ndraw,
                                         burnin=burnin,
                                         how_often=5,
                                         tilt=con.covariance.dot(eta))

            lower_lim, upper_lim = family.equal_tailed_interval(observed, 1 - coverage)

            # in the model we've chosen, the parameter beta is associated
            # to the natural parameter as below
            # exercise: justify this!

            lower_lim_final = eta.dot(con.covariance.dot(eta)) * lower_lim
            upper_lim_final = eta.dot(con.covariance.dot(eta)) * upper_lim

            intervals.append((lower_lim_final, upper_lim_final))

        else:
            _, _, _, family = gibbs_test(con,
                                         initial,
                                         eta,
                                         UMPU=False,
                                         sigma_known=True,
                                         ndraw=ndraw,
                                         burnin=burnin,
                                         how_often=5,
                                         tilt=con.covariance.dot(eta))

            intervals.append((np.nan, np.nan))

        pval = family.cdf(0, observed)
        pval = 2 * min(pval, 1 - pval)
        pvalues.append(pval)

    return zip(L.active, 
               pvalues,
               intervals), randomized_lasso

