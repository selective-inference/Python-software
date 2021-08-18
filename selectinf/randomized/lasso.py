from __future__ import print_function
import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from ..algorithms.sqrt_lasso import solve_sqrt_lasso, choose_lambda

from .query import gaussian_query

from .randomization import randomization
from ..base import (restricted_estimator,
                    _compute_hessian,
                    _pearsonX2)

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

        (self.observed_soln, 
         self.observed_subgrad) = self._solve_randomized_problem(
                                     perturb=perturb, 
                                     solve_args=solve_args)

        active_signs = np.sign(self.observed_soln)
        active = self._active = active_signs != 0

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()

        # don't release sign of unpenalized variables
        _active_signs[unpenalized] = np.nan  
        ordered_variables = list((tuple(np.nonzero(active)[0]) +
                                  tuple(np.nonzero(unpenalized)[0])))
        self.selection_variable = {'sign': _active_signs,
                                   'variables': ordered_variables}

        # initial state for opt variables

        initial_scalings = np.fabs(self.observed_soln[active])
        initial_unpenalized = self.observed_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized])

        _beta_unpenalized = restricted_estimator(self.loglike, 
                                                 self._overall, 
                                                 solve_args=solve_args)

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # form linear part

        num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        # compute part of hessian

        _hessian, _hessian_active, _hessian_unpen  = _compute_hessian(self.loglike,
                                                                      beta_bar,
                                                                      active,
                                                                      unpenalized)
        
        # fill in pieces of query
        
        opt_linear = np.zeros((p, num_opt_var))
        _score_linear_term = np.zeros((p, num_opt_var))

        _score_linear_term = -np.hstack([_hessian_active, _hessian_unpen])

        # set the observed score (data dependent) state

        # observed_score_state is
        # \nabla \ell(\bar{\beta}_E) - Q(\bar{\beta}_E) \bar{\beta}_E
        # in linear regression this is _ALWAYS_ -X^TY
        # 
        # should be asymptotically equivalent to
        # \nabla \ell(\beta^*) - Q(\beta^*)\beta^*

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p, 
                                                          j, 
                                                          active_signs[j]) 
                                      for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = (_hessian_active * active_signs[None, active] 
                            + self.ridge_term * active_directions)

        opt_linear[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), num_opt_var)
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for 
                                           j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            opt_linear[:, unpenalized_slice] = (_hessian_unpen
                                                + self.ridge_term *
                                                unpenalized_directions)

        self.opt_linear = opt_linear
        # now make the constraints and implied gaussian

        self._setup = True
        A_scaling = -np.identity(num_opt_var)
        b_scaling = np.zeros(num_opt_var)

        self._setup_sampler_data = (A_scaling[:active.sum()],
                                    b_scaling[:active.sum()],
                                    opt_linear,
                                    self.observed_subgrad)

        #### to be fixed -- set the cov_score here without dispersion

        self._unscaled_cov_score = _hessian

        #####
        
        if num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data)

        return active_signs

    def _solve_randomized_problem(self, 
                                  perturb=None, 
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 
                                     0, 
                                     -self._initial_omega, 
                                     0)

        problem = rr.simple_problem(self.loglike, self.penalty)

        observed_soln = problem.solve(quad, **solve_args) 
        observed_subgrad = -(self.loglike.smooth_objective(observed_soln, 
                                                          'grad') +
                            quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad

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
        is determined by the Hessian by default,
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

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """

        loglike = rr.glm.gaussian(X, 
                                  Y, 
                                  coef=1. / sigma ** 2, 
                                  quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / (np.sqrt(n - 1) * sigma**2)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y, ddof=1)

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike, 
                     np.asarray(feature_weights) / sigma ** 2,
                     ridge_term, 
                     randomizer)

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

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """
        n, p = X.shape

        loglike = rr.glm.logistic(X, successes, trials=trials, quadratic=quadratic)

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / np.sqrt(n - 1)

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

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """
        n, p = X.shape
        loglike = rr.glm.cox(X, times, status, quadratic=quadratic)

        # scale for randomization seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / np.sqrt(n - 1)

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

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """
        n, p = X.shape
        loglike = rr.glm.poisson(X, counts, quadratic=quadratic)

        # scale for randomizer seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / np.sqrt(n - 1)

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

class split_lasso(lasso):

    """
    Data split, then LASSO (i.e. data carving)
    """

    def __init__(self,
                 loglike,
                 feature_weights,
                 proportion_select,
                 ridge_term=0,
                 perturb=None,
                 estimate_dispersion=False):

        (self.loglike,
         self.feature_weights,
         self.proportion_select,
         self.ridge_term) = (loglike,
                             feature_weights,
                             proportion_select,
                             ridge_term)

        self.nfeature = p = self.loglike.shape[0]
        self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)
        self._initial_omega = perturb
        self.estimate_dispersion = estimate_dispersion

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        signs = lasso.fit(self, 
                          solve_args=solve_args,
                          perturb=perturb)
        
        # for data splitting randomization,
        # we need to estimate a dispersion parameter

        # we then setup up the sampler again

        if self.estimate_dispersion:

            X, y = self.loglike.data
            n, p = X.shape
            df_fit = len(self.selection_variable['variables'])

            dispersion = 2 * (self.loglike.smooth_objective(self._beta_full, 
                                                            'func') /
                          (n - df_fit))

            # run setup again after 
            # estimating dispersion 

            if df_fit > 0:
                self._setup_sampler(*self._setup_sampler_data, 
                                     dispersion=dispersion)

        return signs

    def _setup_implied_gaussian(self, 
                                opt_linear, 
                                observed_subgrad,
                                dispersion=1):

        # key observation is that the covariance of the added noise is 
        # roughly dispersion * (1 - pi) / pi * X^TX (in OLS regression, similar for other
        # models), so the precision is  (X^TX)^{-1} * (pi / ((1 - pi) * dispersion))
        # and prec.dot(opt_linear) = S_E / (dispersion * (1 - pi) / pi)
        # because opt_linear has shape p x E with the columns
        # being those non-zero columns of the solution. Above S_E = np.diag(signs)
        # the conditional precision is S_E Q[E][:,E] * pi / ((1 - pi) * dispersion) S_E
        # and regress_opt is -Q[E][:,E]^{-1} S_E
        # padded with zeros
        # to be E x p

        pi_s = self.proportion_select
        ratio = (1 - pi_s) / pi_s

        ordered_vars = self.selection_variable['variables']
        
        cond_precision = opt_linear[ordered_vars] / (dispersion * ratio)

        signs = self.selection_variable['sign'][ordered_vars]
        signs[np.isnan(signs)] = 1

        cond_precision *= signs[:, None]
        assert(np.linalg.norm(cond_precision - cond_precision.T) / 
               np.linalg.norm(cond_precision) < 1.e-6)
        cond_cov = np.linalg.inv(cond_precision)
        regress_opt = np.zeros((len(ordered_vars),
                                   self.nfeature)) 
        regress_opt[:, ordered_vars] = -cond_cov * signs[None, :] / (dispersion * ratio)
        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        prod_score_prec = np.identity(self.nfeature) / ratio
        
        cov_rand = self._unscaled_cov_score * dispersion

        M1 = prod_score_prec * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T) 
    
        # would be nice to not store these?
        
        self.M1 = M1  
        self.M2 = M2
        self.M3 = M3

        return (cond_mean,
                cond_cov,
                cond_precision,
                regress_opt,
                M1,
                M2,
                M3)

    def _solve_randomized_problem(self, 
                                  # optional binary vector 
                                  # indicating selection data 
                                  perturb=None, 
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if none supplied
        if perturb is not None:
            self._selection_idx = perturb
        if not hasattr(self, "_selection_idx"):
            X, y = self.loglike.data
            total_size = n = X.shape[0]
            pi_s = self.proportion_select
            self._selection_idx = np.zeros(n, np.bool)
            self._selection_idx[:int(pi_s*n)] = True
            np.random.shuffle(self._selection_idx)

        inv_frac = 1 / self.proportion_select
        quad = rr.identity_quadratic(self.ridge_term,
                                     0,
                                     0,
                                     0)
        
        randomized_loss = self.loglike.subsample(self._selection_idx)
        randomized_loss.coef *= inv_frac

        problem = rr.simple_problem(randomized_loss, self.penalty)
        observed_soln = problem.solve(quad, **solve_args) 
        observed_subgrad = -(randomized_loss.smooth_objective(observed_soln,
                                                             'grad') +
                            quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad

    @staticmethod
    def gaussian(X,
                 Y,
                 feature_weights,
                 proportion,
                 sigma=1.,
                 quadratic=None,
                 estimate_dispersion=True):
        r"""
        Squared-error LASSO with feature weights.
        Objective function is (before randomization)

        .. math::

            \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + 
           \sum_{i=1}^p \lambda_i |\beta_i|

        where $\lambda$ is `feature_weights`. The ridge term
        is determined by the Hessian and `np.std(Y)` by default.

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

        proportion: float
            What proportion of data to use for selection.
 
        sigma : float (optional)
            Noise variance. Set to 1 if `covariance_estimator` is not None.
            This scales the loglikelihood by `sigma**(-2)`.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """

        loglike = rr.glm.gaussian(X, 
                                  Y, 
                                  coef=1. / sigma ** 2, 
                                  quadratic=quadratic)

        return split_lasso(loglike, 
                           np.asarray(feature_weights)/sigma**2,
                           proportion,
                           estimate_dispersion=estimate_dispersion)


    @staticmethod
    def logistic(X,
                 successes,
                 feature_weights,
                 proportion,
                 trials=None,
                 quadratic=None):
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

        proportion: float
            What proportion of data to use for selection.
 
        trials : ndarray (optional)
            Number of trials per response, defaults to
            ones the same shape as Y.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """

        loglike = rr.glm.logistic(X,
                                  successes,
                                  trials=trials,
                                  quadratic=quadratic)

        return split_lasso(loglike, 
                           np.asarray(feature_weights),
                           proportion)

    @staticmethod
    def coxph(X,
              times,
              status,
              feature_weights,
              proportion,
              quadratic=None):
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

        proportion: float
            What proportion of data to use for selection.
 
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """
        n, p = X.shape
        loglike = rr.glm.cox(X, times, status, quadratic=quadratic)

        return split_lasso(loglike, 
                           np.asarray(feature_weights),
                           proportion)

    @staticmethod
    def poisson(X,
                counts,
                feature_weights,
                proportion,
                quadratic=None,
                ridge_term=None):
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

        proportion: float
            What proportion of data to use for selection.
 
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.lasso.lasso`

        """
        loglike = rr.glm.poisson(X, counts, quadratic=quadratic)

        return split_lasso(loglike, 
                           np.asarray(feature_weights),
                           proportion)

    
