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

from .query import (gaussian_query,
                    affine_gaussian_sampler)

from .randomization import randomization
from ..base import restricted_estimator
from ..algorithms.debiased_lasso import (debiasing_matrix,
                                         pseudoinverse_debiasing_matrix)

#### High dimensional version
#### - parametric covariance
#### - Gaussian randomization

class lasso(gaussian_query):


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
                 randomizer,
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

        randomizer : object
            Randomizer -- contains representation of randomization density.

        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        self.ridge_term = ridge_term
        self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)
        self._initial_omega = perturb  # random perturbation

        self.randomizer = randomizer

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

        self.initial_soln, self.initial_subgrad = self._solve_randomized_problem(perturb=perturb, solve_args=solve_args)

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

        initial_scalings = np.fabs(self.initial_soln[active])
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized])

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # observed state for score in internal coordinates
        # XXX is this used anywhere?

        self.observed_internal_state = np.hstack([_beta_unpenalized,
                                                  -self.loglike.smooth_objective(beta_bar, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        opt_linear = np.zeros((p, self.num_opt_var))
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
        opt_linear[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), self.num_opt_var)
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            opt_linear[:, unpenalized_slice] = (_hessian_unpen
                                                      + self.ridge_term * unpenalized_directions)

        opt_offset = self.initial_subgrad

        # now make the constraints and implied gaussian

        self._setup = True
        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        self._set_sampler(A_scaling,
                          b_scaling,
                          opt_linear,
                          opt_offset)

        return active_signs

    def _solve_randomized_problem(self, 
                                  perturb=None, 
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)

        initial_soln = problem.solve(quad, **solve_args) 
        initial_subgrad = -(self.loglike.smooth_objective(initial_soln, 'grad') +
                            quad.objective(initial_soln, 'grad'))

        return initial_soln, initial_subgrad

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

        .. math::

            \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + \sum_{i=1}^p \lambda_i |\beta_i|

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

        L : `selection.randomized.lasso.lasso`

        """

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike, 
                     np.asarray(feature_weights) / sigma ** 2,
                     ridge_term, randomizer)

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

        .. math::

             \beta \mapsto \ell(X\beta) + \sum_{i=1}^p \lambda_i |\beta_i|

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

        L : `selection.randomized.lasso.lasso`

        """
        n, p = X.shape

        loglike = rr.glm.logistic(X, successes, trials=trials, quadratic=quadratic)

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike, 
                     np.asarray(feature_weights),
                     ridge_term, randomizer)

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

        .. math::

            \beta \mapsto \ell^{\text{Cox}}(\beta) + 
            \sum_{i=1}^p \lambda_i |\beta_i|

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

        L : `selection.randomized.lasso.lasso`

        """
        loglike = coxph_obj(X, times, status, quadratic=quadratic)

        # scale for randomization seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(times) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer)

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

        .. math::

            \beta \mapsto \ell^{\text{Poisson}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|

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

        L : `selection.randomized.lasso.lasso`

        """
        n, p = X.shape
        loglike = rr.glm.poisson(X, counts, quadratic=quadratic)

        # scale for randomizer seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(counts) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(counts) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer)

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

        .. math::

            \beta \mapsto \|Y-X\beta\|_2 + \sum_{i=1}^p \lambda_i |\beta_i|

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

        L : `selection.randomized.lasso.lasso`

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

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale * denom)

        obj = lasso(loglike, 
                    np.asarray(feature_weights) * denom,
                    ridge_term * denom,
                    randomizer,
                    perturb=perturb * denom)
        obj._sqrt_soln = soln

        return obj

# Targets of inference
# and covariance with score representation

def selected_targets(loglike, 
                     W, 
                     features, 
                     sign_info={}, 
                     dispersion=None,
                     solve_args={'tol': 1.e-12, 'min_its': 50}):

    X, y = loglike.data
    n, p = X.shape

    Xfeat = X[:, features]
    Qfeat = Xfeat.T.dot(W[:, None] * Xfeat)
    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    cov_target = np.linalg.inv(Qfeat)
    _score_linear = -Xfeat.T.dot(W[:, None] * X).T
    crosscov_target_score = _score_linear.dot(cov_target)
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = ((y - loglike.saturated_loss.mean_function(
            Xfeat.dot(observed_target))) ** 2 / W).sum() / (n - Xfeat.shape[1])

    return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

def full_targets(loglike, 
                 W, 
                 features, 
                 dispersion=None,
                 solve_args={'tol': 1.e-12, 'min_its': 50}):

    X, y = loglike.data
    n, p = X.shape
    features_bool = np.zeros(p, np.bool)
    features_bool[features] = True
    features = features_bool

    # target is one-step estimator

    Qfull = X.T.dot(W[:, None] * X)
    Qfull_inv = np.linalg.inv(Qfull)
    full_estimator = loglike.solve(**solve_args)
    cov_target = Qfull_inv[features][:, features]
    observed_target = full_estimator[features]
    crosscov_target_score = np.zeros((p, cov_target.shape[0]))
    crosscov_target_score[features] = -np.identity(cov_target.shape[0])

    if dispersion is None:  # use Pearson's X^2
        dispersion = (((y - loglike.saturated_loss.mean_function(X.dot(full_estimator))) ** 2 / W).sum() / 
                      (n - p))

    alternatives = ['twosided'] * features.sum()
    return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

def debiased_targets(loglike, 
                     W, 
                     features, 
                     sign_info={}, 
                     penalty=None, #required kwarg
                     dispersion=None,
                     approximate_inverse='JM',
                     debiasing_args={}):

    if penalty is None:
        raise ValueError('require penalty for consistent estimator')

    X, y = loglike.data
    n, p = X.shape
    features_bool = np.zeros(p, np.bool)
    features_bool[features] = True
    features = features_bool

    # relevant rows of approximate inverse


    if approximate_inverse == 'JM':
        Qinv_hat = np.atleast_2d(debiasing_matrix(X * np.sqrt(W)[:, None], 
                                                  np.nonzero(features)[0],
                                                  **debiasing_args)) / n
    else:
        Qinv_hat = np.atleast_2d(pseudoinverse_debiasing_matrix(X * np.sqrt(W)[:, None],
                                                                np.nonzero(features)[0],
                                                                **debiasing_args))

    problem = rr.simple_problem(loglike, penalty)
    nonrand_soln = problem.solve()
    G_nonrand = loglike.smooth_objective(nonrand_soln, 'grad')

    observed_target = nonrand_soln[features] - Qinv_hat.dot(G_nonrand)

    if p > n:
        M1 = Qinv_hat.dot(X.T)
        cov_target = (M1 * W[None, :]).dot(M1.T)
        crosscov_target_score = -(M1 * W[None, :]).dot(X).T
    else:
        Qfull = X.T.dot(W[:, None] * X)
        cov_target = Qinv_hat.dot(Qfull.dot(Qinv_hat.T))
        crosscov_target_score = -Qinv_hat.dot(Qfull).T

    if dispersion is None:  # use Pearson's X^2
        Xfeat = X[:, features]
        Qrelax = Xfeat.T.dot(W[:, None] * Xfeat)
        relaxed_soln = nonrand_soln[features] - np.linalg.inv(Qrelax).dot(G_nonrand[features])
        dispersion = (((y - loglike.saturated_loss.mean_function(Xfeat.dot(relaxed_soln)))**2 / W).sum() / 
                      (n - features.sum()))

    alternatives = ['twosided'] * features.sum()
    return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

def form_targets(target, 
                 loglike, 
                 W, 
                 features, 
                 **kwargs):
    _target = {'full':full_targets,
               'selected':selected_targets,
               'debiased':debiased_targets}[target]
    return _target(loglike,
                   W,
                   features,
                   **kwargs)

class split_lasso(lasso):

    """
    Data split, then LASSO (i.e. data carving)
    """

    def __init__(self,
                 loglike,
                 feature_weights,
                 proportion):

        (self.loglike,
         self.feature_weights,
         self.proportion) = (loglike,
                             feature_weights,
                             proportion)

    def _setup_implied_gaussian(self):

        _, prec = self.randomizer.cov_prec
        opt_linear, opt_offset = self.opt_transform

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            logdens_linear = cond_cov.dot(opt_linear.T) * prec
        else:
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            logdens_linear = cond_cov.dot(opt_linear.T).dot(prec)

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        return cond_mean, cond_cov, cond_precision, logdens_linear

    def _solve_randomized_problem(self, 
                                  perturb=None, # optional binary vector indicating selection data 
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb is not None:
            self._selection_idx = perturb
        if self._selection_idx is None:
            X, y = self.loglike.data
            total_size = n = X.shape[0]
            subsample = np.random.binomial(1, self.proportion, size=(n,))
        n, m = self.total_size, self.subsample_size
        inv_frac = n / m
        quadratic = rr.identity_quadratic(epsilon, 0, 0, 0)
        m, n = self.subsample_size, self.total_size # shorthand
        idx = np.zeros(n, np.bool)
        idx[:m] = 1
        np.random.shuffle(idx)

        randomized_loss = loss.subsample(idx)
        randomized_loss.coef *= inv_frac

        randomized_loss.quadratic = quadratic



        problem = rr.simple_problem(self.loglike, self.penalty)

        initial_soln = problem.solve(quad, **solve_args) 
        initial_subgrad = -(self.loglike.smooth_objective(initial_soln, 'grad') +
                            quad.objective(initial_soln, 'grad'))

        initial_soln = problem.solve(quad, **solve_args)         
        return initial_soln, initial_subgrad

