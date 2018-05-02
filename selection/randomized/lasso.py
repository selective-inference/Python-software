from __future__ import print_function
import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr
import regreg.affine as ra

from ..constraints.affine import constraints
from ..algorithms.sqrt_lasso import solve_sqrt_lasso, choose_lambda

from .query import (query,
                    multiple_queries,
                    langevin_sampler,
                    affine_gaussian_sampler)

from .reconstruction import reconstruct_opt
from .randomization import randomization
from .base import restricted_estimator
from .glm import (pairs_bootstrap_glm,
                  glm_nonparametric_bootstrap,
                  glm_parametric_covariance)
from ..algorithms.debiased_lasso import debiasing_matrix

#### High dimensional version
#### - parametric covariance
#### - Gaussian randomization

class lasso(object):
    r"""
    A class for the randomized LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta) + 
            \sum_{i=1}^p \lambda_i |\beta_i\| - \omega^T\beta + \frac{\epsilon}{2} \|\beta\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty. Each static method
    forms $\ell$ as well as the $\ell_1$ penalty. The generic class
    forms the remaining two terms in the objective.

    """

    def __init__(self,
                 loglike,
                 feature_weights,
                 ridge_term,
                 randomizer_scale,
                 perturb=None):
        r"""
        Create a new post-selection object for the LASSO problem
        Parameters
        ----------
        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.
        feature_weights : np.ndarray
            Feature weights for L-1 penalty. If a float,
            it is brodcast to all features.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomization.
        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        self.ridge_term = ridge_term
        self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)
        self._initial_omega = perturb  # random perturbation

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):
        """
        Fit the randomized lasso using `regreg`.
        Parameters
        ----------
        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.
        Returns
        -------
        signs : np.float
             Support and non-zero signs of randomized lasso solution.

        """

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        self.initial_soln = problem.solve(quad, **solve_args)

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan  # don't release sign of unpenalized variables
        self.selection_variable = {'sign': _active_signs,
                                   'variables': self._overall}

        # initial state for opt variables

        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') +
                            quad.objective(self.initial_soln, 'grad'))
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.fabs(self.initial_soln[active])
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized])

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # observed state for score in internal coordinates

        self.observed_internal_state = np.hstack([_beta_unpenalized,
                                                  -self.loglike.smooth_objective(beta_bar, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self.num_opt_var))
        _score_linear_term = np.zeros((p, self.num_opt_var))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        _hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _hessian_unpen = np.dot(X.T, X[:, unpenalized] * W[:, None])

        _score_linear_term = -np.hstack([_hessian_active, _hessian_unpen])

        # set the observed score (data dependent) state

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p, j, active_signs[j]) for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = _hessian_active * active_signs[None, active] + self.ridge_term * active_directions
        _opt_linear_term[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), self.num_opt_var)
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (_hessian_unpen
                                                      + self.ridge_term * unpenalized_directions)

        # two transforms that encode score and optimization
        # variable roles

        self.opt_transform = (_opt_linear_term, self.initial_subgrad)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self._setup = True
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.ndim = self.loglike.shape[0]

        # compute implied mean and covariance

        cov, prec = self.randomizer.cov_prec
        opt_linear, opt_offset = self.opt_transform

        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        def log_density(logdens_linear, offset, cond_prec, score, opt):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

        log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

        # now make the constraints

        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        affine_con = constraints(A_scaling,
                                 b_scaling,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        logdens_transform = (logdens_linear, opt_offset)

        self.sampler = affine_gaussian_sampler(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               logdens_transform,
                                               selection_info=self.selection_variable)  # should be signs and the subgradients we've conditioned on

        return active_signs

    def summary(self,
                target="selected",
                features=None,
                parameter=None,
                level=0.9,
                ndraw=10000,
                burnin=2000,
                compute_intervals=False,
                dispersion=None):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        target : one of ['selected', 'full']
        features : np.bool
            Binary encoding of which features to use in final
            model and targets.
        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.
        level : float
            Confidence level.
        ndraw : int (optional)
            Defaults to 1000.
        burnin : int (optional)
            Defaults to 1000.
        compute_intervals : bool
            Compute confidence intervals?
        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?
        """

        if parameter is None:
            parameter = np.zeros(self.loglike.shape[0])

        if target == 'selected':
            observed_target, cov_target, cov_target_score, alternatives = self.selected_targets(features=features,
                                                                                                dispersion=dispersion)
        else:
            X, y = self.loglike.data
            n, p = X.shape
            if n > p and target == 'full':
                observed_target, cov_target, cov_target_score, alternatives = self.full_targets(features=features,
                                                                                                dispersion=dispersion)
            else:
                observed_target, cov_target, cov_target_score, alternatives = self.debiased_targets(features=features,
                                                                                                    dispersion=dispersion)

        if self._overall.sum() > 0:
            opt_sample = self.sampler.sample(ndraw, burnin)

            pivots = self.sampler.coefficient_pvalues(observed_target,
                                                      cov_target,
                                                      cov_target_score,
                                                      parameter=parameter,
                                                      sample=opt_sample,
                                                      alternatives=alternatives)
            if not np.all(parameter == 0):
                pvalues = self.sampler.coefficient_pvalues(observed_target,
                                                           cov_target,
                                                           cov_target_score,
                                                           parameter=np.zeros_like(parameter),
                                                           sample=opt_sample,
                                                           alternatives=alternatives)
            else:
                pvalues = pivots

            intervals = None
            if compute_intervals:
                intervals = self.sampler.confidence_intervals(observed_target,
                                                              cov_target,
                                                              cov_target_score,
                                                              sample=opt_sample)

            return pivots, pvalues, intervals
        else:
            return [], [], []

    def selective_MLE(self,
                      target="selected",
                      features=None,
                      parameter=None,
                      level=0.9,
                      compute_intervals=False,
                      dispersion=None,
                      solve_args={'tol':1.e-12}):
        """
        Parameters
        ----------
        target : one of ['selected', 'full']
        features : np.bool
            Binary encoding of which features to use in final
            model and targets.
        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.
        level : float
            Confidence level.
        ndraw : int (optional)
            Defaults to 1000.
        burnin : int (optional)
            Defaults to 1000.
        compute_intervals : bool
            Compute confidence intervals?
        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?
        """

        if parameter is None:
            parameter = np.zeros(self.loglike.shape[0])

        if target == 'selected':
            observed_target, cov_target, cov_target_score, alternatives = self.selected_targets(features=features,
                                                                                                dispersion=dispersion)
        elif target == 'full':
            X, y = self.loglike.data
            n, p = X.shape
            if n > p:
                observed_target, cov_target, cov_target_score, alternatives = self.full_targets(features=features,
                                                                                                dispersion=dispersion)
            else:
                observed_target, cov_target, cov_target_score, alternatives = self.debiased_targets(features=features,
                                                                                                    dispersion=dispersion)

        # working out conditional law of opt variables given
        # target after decomposing score wrt target

        return self.sampler.selective_MLE(observed_target,
                                          cov_target,
                                          cov_target_score,
                                          self.observed_opt_state,
                                          solve_args=solve_args)

    # Targets of inference
    # and covariance with score representation

    def selected_targets(self, features=None, dispersion=None):

        X, y = self.loglike.data
        n, p = X.shape

        if features is None:
            active = self._active
            unpenalized = self._unpenalized
            noverall = active.sum() + unpenalized.sum()
            overall = active + unpenalized

            score_linear = self.score_transform[0]
            Q = -score_linear[overall]
            cov_target = np.linalg.inv(Q)
            observed_target = self._beta_full[overall]
            crosscov_target_score = score_linear.dot(cov_target)
            Xfeat = X[:, overall]
            alternatives = ([{1: 'greater', -1: 'less'}[int(s)] for s in self.selection_variable['sign'][active]] + 
                            ['twosided'] * unpenalized.sum())

        else:

            features_b = np.zeros_like(self._overall)
            features_b[features] = True
            features = features_b

            Xfeat = X[:, features]
            Qfeat = Xfeat.T.dot(self._W[:, None] * Xfeat)
            Gfeat = self.loglike.smooth_objective(self.initial_soln, 'grad')[features]
            Qfeat_inv = np.linalg.inv(Qfeat)
            one_step = self.initial_soln[features] - Qfeat_inv.dot(Gfeat)
            cov_target = Qfeat_inv
            _score_linear = -Xfeat.T.dot(self._W[:, None] * X).T
            crosscov_target_score = _score_linear.dot(cov_target)
            observed_target = one_step
            alternatives = ['twosided'] * features.sum()

        if dispersion is None:  # use Pearson's X^2
            dispersion = ((y - self.loglike.saturated_loss.mean_function(
                Xfeat.dot(observed_target))) ** 2 / self._W).sum() / (n - Xfeat.shape[1])

        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    def full_targets(self, features=None, dispersion=None):

        if features is None:
            features = self._overall
        features_bool = np.zeros(self._overall.shape, np.bool)
        features_bool[features] = True
        features = features_bool

        X, y = self.loglike.data
        n, p = X.shape

        # target is one-step estimator

        Qfull = X.T.dot(self._W[:, None] * X)
        G = self.loglike.smooth_objective(self.initial_soln, 'grad')
        Qfull_inv = np.linalg.inv(Qfull)
        one_step = self.initial_soln - Qfull_inv.dot(G)
        cov_target = Qfull_inv[features][:, features]
        observed_target = one_step[features]
        crosscov_target_score = np.zeros((p, cov_target.shape[0]))
        crosscov_target_score[features] = -np.identity(cov_target.shape[0])

        if dispersion is None:  # use Pearson's X^2
            dispersion = ((y - self.loglike.saturated_loss.mean_function(X.dot(one_step))) ** 2 / self._W).sum() / (
            n - p)

        alternatives = ['twosided'] * features.sum()
        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    def debiased_targets(self,
                         features=None,
                         dispersion=None,
                         debiasing_args={}):

        if features is None:
            features = self._overall
        features_bool = np.zeros(self._overall.shape, np.bool)
        features_bool[features] = True
        features = features_bool

        X, y = self.loglike.data
        n, p = X.shape

        # target is one-step estimator

        G = self.loglike.smooth_objective(self.initial_soln, 'grad')
        Qinv_hat = np.atleast_2d(debiasing_matrix(X * np.sqrt(self._W)[:, None],
                                                  np.nonzero(features)[0],
                                                  **debiasing_args)) / n
        observed_target = self.initial_soln[features] - Qinv_hat.dot(G)
        if p > n:
            M1 = Qinv_hat.dot(X.T)
            cov_target = (M1 * self._W[None, :]).dot(M1.T)
            crosscov_target_score = -(M1 * self._W[None, :]).dot(X).T
        else:
            Qfull = X.T.dot(self._W[:, None] * X)
            cov_target = Qinv_hat.dot(Qfull.dot(Qinv_hat.T))
            crosscov_target_score = -Qinv_hat.dot(Qfull).T

        if dispersion is None:  # use Pearson's X^2
            Xfeat = X[:, features]
            Qrelax = Xfeat.T.dot(self._W[:, None] * Xfeat)
            relaxed_soln = self.initial_soln[features] - np.linalg.inv(Qrelax).dot(G[features])
            dispersion = ((y - self.loglike.saturated_loss.mean_function(
                Xfeat.dot(relaxed_soln))) ** 2 / self._W).sum() / (n - features.sum())
        alternatives = ['twosided'] * features.sum()
        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    @staticmethod
    def gaussian(X,
                 Y,
                 feature_weights,
                 sigma=1.,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None):
        r"""
        Squared-error LASSO with feature weights.
        Objective function is (before randomization)

        $$
        \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        $$

        where $\lambda$ is `feature_weights`. The ridge term
        is determined by the Hessian and `np.std(Y)` by default,
        as is the randomizer scale.
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
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        return lasso(loglike, np.asarray(feature_weights) / sigma ** 2,
                       ridge_term, randomizer_scale)

    @staticmethod
    def logistic(X,
                 successes,
                 feature_weights,
                 trials=None,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None):
        r"""
        Logistic LASSO with feature weights (before randomization)
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
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        n, p = X.shape

        loglike = rr.glm.logistic(X, successes, trials=trials, quadratic=quadratic)

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5

        return lasso(loglike, np.asarray(feature_weights),
                       ridge_term, randomizer_scale)

    @staticmethod
    def coxph(X,
              times,
              status,
              feature_weights,
              quadratic=None,
              ridge_term=None,
              randomizer_scale=None):
        r"""
        Cox proportional hazards LASSO with feature weights.
        Objective function is (before randomization)

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
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        loglike = coxph_obj(X, times, status, quadratic=quadratic)

        # scale for randomization seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(times) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer_scale)

    @staticmethod
    def poisson(X,
                counts,
                feature_weights,
                quadratic=None,
                ridge_term=None,
                randomizer_scale=None):
        r"""
        Poisson log-linear LASSO with feature weights.
        Objective function is (before randomization)

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
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        n, p = X.shape
        loglike = rr.glm.poisson(X, counts, quadratic=quadratic)

        # scale for randomizer seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(counts) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(counts) * np.sqrt(n / (n - 1.))

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer_scale)

    @staticmethod
    def sqrt_lasso(X,
                   Y,
                   feature_weights,
                   quadratic=None,
                   ridge_term=None,
                   randomizer_scale=None,
                   solve_args={'min_its': 200},
                   perturb=None):
        r"""
        Use sqrt-LASSO to choose variables.
        Objective function is (before randomization)

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
        solve_args : dict
            Arguments passed to solver.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

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

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / (n - 1)

        if randomizer_scale is None:
            randomizer_scale = 0.5 * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if perturb is None:
            perturb = np.random.standard_normal(p) * randomizer_scale

        randomQ = rr.identity_quadratic(ridge_term, 0, -perturb, 0)  # a ridge + linear term

        if quadratic is not None:
            totalQ = randomQ + quadratic
        else:
            totalQ = randomQ

        soln, sqrt_loss = solve_sqrt_lasso(X,
                                           Y,
                                           weights=feature_weights,
                                           quadratic=totalQ,
                                           solve_args=solve_args,
                                           force_fat=True)

        denom = np.linalg.norm(Y - X.dot(soln))
        loglike = rr.glm.gaussian(X, Y)

        obj = lasso(loglike, np.asarray(feature_weights) * denom,
                    ridge_term * denom,
                    randomizer_scale * denom,
                    perturb=perturb * denom)
        obj._sqrt_soln = soln

        return obj
