"""
Classes encapsulating some common workflows in randomized setting
"""

from copy import copy

import numpy as np
import regreg.api as rr

from .glm import (glm_group_lasso,
                  glm_group_lasso_parametric,
                  glm_greedy_step,
                  glm_threshold_score,
                  glm_nonparametric_bootstrap,
                  glm_parametric_covariance,
                  pairs_bootstrap_glm)
from .randomization import randomization
from .query import multiple_queries
from .M_estimator import restricted_Mest

class lasso(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \frac{1}{2n} \|y-X\beta\|^2_2 + 
            \lambda \|\beta\|_1 - \omega^T\beta + \frac{\epsilon}{2} \|\beta\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty.

    """


    def __init__(self, 
                 loglike, 
                 feature_weights,
                 ridge_term,
                 randomizer_scale,
                 randomizer='gaussian',
                 parametric_cov_estimator=False):
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

        randomizer : str (optional)
            One of ['laplace', 'logistic', 'gaussian']


        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        self.parametric_cov_estimator = parametric_cov_estimator

        if randomizer == 'laplace':
            self.randomizer = randomization.laplace((p,), scale=randomizer_scale)
        elif randomizer == 'gaussian':
            self.randomizer = randomization.isotropic_gaussian((p,),randomizer_scale)
        elif randomizer == 'logistic':
            self.randomizer = randomization.logistic((p,), scale=randomizer_scale)

        self.ridge_term = ridge_term

        self.penalty = rr.group_lasso(np.arange(p),
                                      weights=dict(zip(np.arange(p), self.feature_weights)), lagrange=1.)

    def fit(self, solve_args={'tol':1.e-12, 'min_its':50}, 
            views=[], nboot=1000):
        """
        Fit the randomized lasso using `regreg`.

        Parameters
        ----------

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        views : list
             Other views of the data, e.g. cross-validation.

        Returns
        -------

        sign_beta : np.float
             Support and non-zero signs of randomized lasso solution.
             
        """

        p = self.nfeature
        if self.parametric_cov_estimator==True:
            self._view = glm_group_lasso_parametric(self.loglike, self.ridge_term, self.penalty, self.randomizer)
        else:
            self._view = glm_group_lasso(self.loglike, self.ridge_term, self.penalty, self.randomizer)
        self._view.solve(nboot=nboot)

        views = copy(views); views.append(self._view)
        self._queries = multiple_queries(views)
        self._queries.solve()
   
        self.signs = np.sign(self._view.initial_soln)
        self.selection_variable = self._view.selection_variable
        return self.signs

    def decompose_subgradient(self,
                              conditioning_groups=None,
                              marginalizing_groups=None):
        """

        Marginalize over some if inactive part of subgradient
        if applicable.

        Parameters
        ----------

        conditioning_groups : np.bool
             Which groups' subgradients should we condition on.

        marginalizing_groups : np.bool
             Which groups' subgradients should we marginalize over.

        Returns
        -------

        None

        """

        if not hasattr(self, "_view"):
            raise ValueError("fit method should be run first")
        self._view.decompose_subgradient(conditioning_groups=conditioning_groups, 
                                         marginalizing_groups=marginalizing_groups)

    def summary(self,
                selected_features,
                parameter=None,
                level=0.9,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=False,
                bootstrap_sampler=False):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
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

        bootstrap : bool
            Use wild bootstrap instead of Gaussian plugin.

        """
        if not hasattr(self, "_queries"):
            raise ValueError('run `fit` method before producing summary.')

        if parameter is None:
            parameter = np.zeros(self.loglike.shape[0])

        unpenalized_mle = restricted_Mest(self.loglike, selected_features)

        if self.parametric_cov_estimator == False:
            n = self.loglike.data[0].shape[0]
            form_covariances = glm_nonparametric_bootstrap(n, n)
            boot_target, boot_target_observed = pairs_bootstrap_glm(self.loglike, selected_features, inactive=None)
            target_info = boot_target
        else:
            target_info = (selected_features, np.identity(unpenalized_mle.shape[0]))
            form_covariances = glm_parametric_covariance(self.loglike)

        opt_samplers = []
        for q in self._queries.objectives:
            cov_info = q.setup_sampler()
            if self.parametric_cov_estimator == False:
                target_cov, score_cov = form_covariances(target_info,  
                                                         cross_terms=[cov_info],
                                                         nsample=q.nboot)
            else:
                target_cov, score_cov = form_covariances(target_info,  
                                                         cross_terms=[cov_info])

            opt_samplers.append(q.sampler)

        opt_samples = [opt_sampler.sample(ndraw,
                                          burnin) for opt_sampler in opt_samplers]

        ### TODO -- this only uses one view -- what about other queries?

        pivots = opt_samplers[0].coefficient_pvalues(unpenalized_mle, target_cov, score_cov, parameter=parameter, sample=opt_samples[0])
        if not np.all(parameter == 0):
            pvalues = opt_samplers[0].coefficient_pvalues(unpenalized_mle, target_cov, score_cov, parameter=np.zeros_like(parameter), sample=opt_samples[0])
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = opt_samplers[0].confidence_intervals(unpenalized_mle, target_cov, score_cov, sample=opt_samples[0])

        return pivots, pvalues, intervals

    @staticmethod
    def gaussian(X, 
                 Y, 
                 feature_weights, 
                 sigma=1.,
                 parametric_cov_estimator=False,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None,
                 randomizer='gaussian'):
        r"""
        Squared-error LASSO with feature weights.

        Objective function (before randomizer) is 
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

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma**2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X**2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y)**2 * mean_diag / np.sqrt(n)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y)

        return lasso(loglike, np.asarray(feature_weights) / sigma**2,
                     ridge_term, randomizer_scale, randomizer=randomizer,
                     parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def logistic(X, 
                 successes, 
                 feature_weights, 
                 trials=None,
                 parametric_cov_estimator=False,
                 quadratic=None,
                 ridge_term=None,
                 randomizer='gaussian',
                 randomizer_scale=None):
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

        mean_diag = np.mean((X**2).sum(0))

        if ridge_term is None:
            ridge_term = mean_diag / np.sqrt(n)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 

        return lasso(loglike, feature_weights, 
                     ridge_term, 
                     randomizer_scale,
                     parametric_cov_estimator=parametric_cov_estimator,
                     randomizer=randomizer)

    @staticmethod
    def coxph(X, 
              times, 
              status, 
              feature_weights,
              parametric_cov_estimator=False,
              quadratic=None,
              ridge_term=None,
              randomizer='gaussian',
              randomizer_scale=None):
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

        mean_diag = np.mean((X**2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(Y)**2 * mean_diag / np.sqrt(n)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y)

        return lasso(loglike, 
                     feature_weights, 
                     ridge_term,
                     randomizer_scale, 
                     randomizer=randomizer,
                     parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def poisson(X, 
                counts, 
                feature_weights,
                parametric_cov_estimator=False,
                quadratic=None,
                ridge_term=None,
                randomizer_scale=None,
                randomizer='gaussian'):
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

        mean_diag = np.mean((X**2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(counts)**2 * mean_diag / np.sqrt(n)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(counts)

        return lasso(loglike, 
                     feature_weights, 
                     ridge_term,
                     randomizer_scale, 
                     randomizer=randomizer,
                     parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def sqrt_lasso(X, 
                   Y, 
                   feature_weights, 
                   quadratic=None,
                   parametric_cov_estimator=False,
                   sigma_estimate='truncated',
                   solve_args={'min_its':200},
                   randomizer_scale=None,
                   randomizer='gaussian'):
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

        raise NotImplementedError

        n, p = X.shape

        # scale for randomization seems kind of meaningless here...

        mean_diag = np.mean((X**2).sum(0))
        ridge_term = np.std(Y)**2 * mean_diag / np.sqrt(n)
        randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y)

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

        loglike = rr.glm.gaussian(X, Y, quadratic=quadratic)

        L = lasso(loglike, feature_weights * multiplier * sigma_E,
                  parametric_cov_estimator=parametric_cov_estimator,
                  ignore_inactive_constraints=True)

        # these arguments are reused for data carving

        if nactive:
            L._sigma_hat = _sigma_hat
            L._sigma_estimator_args = _sigma_estimator_args
            L._weight_multiplier = multiplier * sigma_E
            L._multiplier = multiplier
            L.lasso_solution = soln

        return L


class step(lasso):

    r"""
    A class for maximizing some coordinates of the
    randomized score of a GLM. The problem we are
    solving is

    .. math::

        \text{minimize}_{\eta} (\nabla \ell(\bar{\beta}_E) - \omega)^T\eta

    subject to $\|\eta_g\|_2/w_g \leq 1$ where $w_g$ are group weights.
    The set of variables $E$ are variables we have partially maximized over
    and $\bar{\beta}_E$ should be viewed as padded out with zeros
    over all variables in $E^c$.

    """


    def __init__(self, 
                 loglike, 
                 feature_weights,
                 candidate,
                 randomizer_scale,
                 active=None,
                 randomizer='gaussian',
                 parametric_cov_estimator=False):
        r"""

        Create a new post-selection for the stepwise problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        feature_weights : np.ndarray
            Feature weights for L-1 penalty. If a float,
            it is brodcast to all features.

        candidate : np.bool
            Which groups of variables are candidates
            for inclusion in this step.

        randomizer_scale : float
            Scale for IID components of randomization.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.

        randomizer : str (optional)
            One of ['laplace', 'logistic', 'gaussian']


        """

        self.active = active
        self.candidate = candidate

        self.loglike = loglike
        self.nfeature = p = loglike.shape[0]

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        self.parametric_cov_estimator = parametric_cov_estimator

        nrandom = candidate.sum()
        if randomizer == 'laplace':
            self.randomizer = randomization.laplace((nrandom,), scale=randomizer_scale)
        elif randomizer == 'gaussian':
            self.randomizer = randomization.isotropic_gaussian((nrandom,),randomizer_scale)
        elif randomizer == 'logistic':
            self.randomizer = randomization.logistic((nrandom,), scale=randomizer_scale)

        self.penalty = rr.group_lasso(np.arange(p),
                                      weights=dict(zip(np.arange(p), self.feature_weights)), lagrange=1.)

    def fit(self, 
            views=[]):
        """
        Find the maximizing group.

        Parameters
        ----------

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        views : list
             Other views of the data, e.g. cross-validation.

        Returns
        -------

        sign_beta : np.float
             Support and non-zero signs of randomized lasso solution.
             
        """

        p = self.nfeature
        self._view = glm_greedy_step(self.loglike, 
                                     self.penalty, 
                                     self.active,
                                     self.candidate,
                                     self.randomizer)
        self._view.solve()

        views = copy(views); views.append(self._view)
        self._queries = multiple_queries(views)
        self._queries.solve()
   
        self.maximizing_group = self._view.selection_variable['maximizing_group']
        return self.maximizing_group

    def decompose_subgradient(self,
                              conditioning_groups=None,
                              marginalizing_groups=None):
        """

        Marginalize over some if candidate part of subgradient
        if applicable.

        Parameters
        ----------

        conditioning_groups : np.bool
             Which groups' subgradients should we condition on.

        marginalizing_groups : np.bool
             Which groups' subgradients should we marginalize over.

        Returns
        -------

        None

        """
        raise NotImplementedError

    @staticmethod
    def gaussian(X, 
                 Y, 
                 feature_weights, 
                 candidate=None,
                 active=None,
                 randomizer_scale=None,
                 parametric_cov_estimator=False,
                 randomizer='gaussian'):
        r"""
        Take a step with a Gaussian loglikelihood.

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

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.step`
        

        """
        loglike = rr.glm.gaussian(X, Y)
        n, p = X.shape

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        if randomizer_scale is None:
            mean_diag = np.mean((X**2).sum(0))
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y)

        return step(loglike, 
                    feature_weights,
                    candidate, 
                    randomizer_scale, 
                    active=active,
                    randomizer=randomizer,
                    parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def logistic(X, 
                 successes, 
                 feature_weights, 
                 active=None,
                 candidate=None,
                 trials=None, 
                 parametric_cov_estimator=False,
                 randomizer_scale=None,
                 randomizer='gaussian'):
        r"""
        Take a step with a logistic loglikelihood.

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

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        trials : ndarray (optional)
            Number of trials per response, defaults to
            ones the same shape as Y. 

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.step`

        """
        n, p = X.shape
        loglike = rr.glm.logistic(X, successes, trials=trials)

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        if randomizer_scale is None:
            mean_diag = np.mean((X**2).sum(0))
            randomizer_scale = np.sqrt(mean_diag) * 0.5 

        return step(loglike, 
                    feature_weights, 
                    candidate,
                    randomizer_scale,
                    active=active,
                    parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def coxph(X, 
              times, 
              status, 
              feature_weights, 
              candidate=None,
              active=None,
              parametric_cov_estimator=False,
              randomizer_scale=None,
              randomizer='gaussian'):
        r"""
        Take a step with a Cox partial loglikelihood.

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

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.lasso`
        

        """
        n, p = X.shape
        loglike = coxph_obj(X, times, status)

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        if randomizer_scale is None:
            randomizer_scale = 1. / np.sqrt(n)

        return step(loglike, 
                    feature_weights, 
                    candidate,
                    randomizer_scale,
                    active=active,
                    randomizer=randomizer,
                    parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def poisson(X, 
                counts, 
                feature_weights, 
                candidate=None,
                active=None,
                parametric_cov_estimator=False,
                randomizer_scale=None,
                randomizer='gaussian'):
        r"""
        Take a step with a Poisson loglikelihood.

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

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.step`
        

        """
        n, p = X.shape
        loglike = rr.glm.poisson(X, counts)

        # scale for randomizer seems kind of meaningless here...

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        mean_diag = np.mean((X**2).sum(0))
        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(counts)

        return step(loglike, 
                    feature_weights, 
                    candidate,
                    randomizer_scale, 
                    active=active,
                    randomizer=randomizer,
                    parametric_cov_estimator=parametric_cov_estimator)

class threshold(lasso):

    r"""
    A class for thresholding some coordinates of the
    randomized score of a GLM. The problem we are
    solving is

    .. math::

        \text{minimize}_{\eta: |\eta_i| \leq \tau_i} \frac{1}{2}\|\nabla \ell(\bar{\beta}_E) + \omega - \eta\|^2_2

    The set of variables $E$ are variables we have partially maximized over
    and $\bar{\beta}_E$ should be viewed as padded out with zeros
    over all variables in $E^c$.

    """

    def __init__(self, 
                 loglike, 
                 threshold_value,
                 candidate,
                 randomizer_scale,
                 active=None,
                 randomizer='gaussian',
                 parametric_cov_estimator=False):
        r"""

        Create a new post-selection for the stepwise problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        threshold_value : [float, sequence]
            Thresholding for each feature. If 1d defaults
            it is treated as a multiple of np.ones.

        candidate : np.bool
            Which groups of variables are candidates
            for thresholding.

        randomizer_scale : float
            Scale for IID components of randomization.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.

        randomizer : str (optional)
            One of ['laplace', 'logistic', 'gaussian']

        """

        self.active = active
        self.candidate = candidate

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(threshold_value).shape == ():
            threshold = np.ones(loglike.shape) * threshold_value
        self.threshold_value = np.asarray(threshold_value)[self.candidate]

        self.parametric_cov_estimator = parametric_cov_estimator

        nrandom = candidate.sum()
        if randomizer == 'laplace':
            self.randomizer = randomization.laplace((nrandom,), scale=randomizer_scale)
        elif randomizer == 'gaussian':
            self.randomizer = randomization.isotropic_gaussian((nrandom,),randomizer_scale)
        elif randomizer == 'logistic':
            self.randomizer = randomization.logistic((nrandom,), scale=randomizer_scale)

    def fit(self, 
            views=[]):
        """
        Find the maximizing group.

        Parameters
        ----------

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        views : list
             Other views of the data, e.g. cross-validation.

        Returns
        -------

        sign_beta : np.float
             Support and non-zero signs of randomized lasso solution.
             
        """

        p = self.nfeature
        self._view = glm_threshold_score(self.loglike, 
                                         self.threshold_value,
                                         self.randomizer,
                                         self.active,
                                         self.candidate)
        self._view.solve()

        views = copy(views); views.append(self._view)
        self._queries = multiple_queries(views)
        self._queries.solve()
   
        self.boundary = self._view.selection_variable['boundary_set']
        return self.boundary

    def decompose_subgradient(self,
                              conditioning_groups=None,
                              marginalizing_groups=None):
        """

        Marginalize over some if candidate part of subgradient
        if applicable.

        Parameters
        ----------

        conditioning_groups : np.bool
             Which groups' subgradients should we condition on.

        marginalizing_groups : np.bool
             Which groups' subgradients should we marginalize over.

        Returns
        -------

        None

        """
        raise NotImplementedError

    @staticmethod
    def gaussian(X, 
                 Y, 
                 threshold_value, 
                 candidate=None,
                 active=None,
                 parametric_cov_estimator=False,
                 randomizer_scale=None,
                 randomizer='gaussian'):
        r"""
        Take a step with a Gaussian loglikelihood.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        Y : ndarray
            Shape (n,) -- the response.

        threshold_value : [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `threshold` to 0. If `threshold` is 
            a float, then all parameters are penalized equally.

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.threshold`
        
        """

        loglike = rr.glm.gaussian(X, Y)
        n, p = X.shape

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        if randomizer_scale is None:
            mean_diag = np.mean((X**2).sum(0))
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y)

        return threshold(loglike, 
                         threshold_value,
                         candidate, 
                         randomizer_scale, 
                         active=active,
                         randomizer=randomizer,
                         parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def logistic(X, 
                 successes, 
                 threshold_value, 
                 active=None,
                 candidate=None,
                 trials=None, 
                 parametric_cov_estimator=False,
                 randomizer_scale=None,
                 randomizer='gaussian'):
        r"""
        Take a step with a logistic loglikelihood.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        successes : ndarray
            Shape (n,) -- response vector. An integer number of successes.
            For data that is proportions, multiply the proportions
            by the number of trials first.

        threshold_value : [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `threshold` to 0. If `threshold` is 
            a float, then all parameters are penalized equally.

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        trials : ndarray (optional)
            Number of trials per response, defaults to
            ones the same shape as Y. 

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.threshold`
        
        """
        n, p = X.shape
        loglike = rr.glm.logistic(X, successes, trials=trials)

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        if randomizer_scale is None:
            mean_diag = np.mean((X**2).sum(0))
            randomizer_scale = np.sqrt(mean_diag) * 0.5 

        return threshold(loglike, 
                         threshold_value,
                         candidate,
                         randomizer_scale,
                         active=active,
                         parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def coxph(X, 
              times, 
              status, 
              threshold_value,
              candidate=None,
              active=None,
              parametric_cov_estimator=False,
              randomizer_scale=None,
              randomizer='gaussian'):
        r"""
        Take a step with a Cox partial loglikelihood.

        Uses Efron's tie breaking method.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        times : ndarray
            Shape (n,) -- the survival times.

        status : ndarray
            Shape (n,) -- the censoring status.

        threshold_value : [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `threshold` to 0. If `threshold` is 
            a float, then all parameters are penalized equally.

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.threshold`

        """
        n, p = X.shape
        loglike = coxph_obj(X, times, status)

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        if randomizer_scale is None:
            randomizer_scale = 1. / np.sqrt(n)

        return threshold(loglike, 
                         threshold_value,
                         candidate,
                         randomizer_scale,
                         active=active,
                         randomizer=randomizer,
                         parametric_cov_estimator=parametric_cov_estimator)

    @staticmethod
    def poisson(X, 
                counts, 
                threshold_value,
                candidate=None,
                active=None,
                parametric_cov_estimator=False,
                randomizer_scale=None,
                randomizer='gaussian'):
        r"""
        Take a step with a Poisson loglikelihood.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        counts : ndarray
            Shape (n,) -- the response.

        threshold_value : [float, sequence]
            Penalty weights. An intercept, or other unpenalized 
            features are handled by setting those entries of 
            `threshold` to 0. If `threshold` is 
            a float, then all parameters are penalized equally.

        candidate : np.bool (optional)
            Which groups of variables are candidates
            for inclusion in this step. Defaults to ~active.

        active : np.bool (optional)
            Which groups of variables make up $E$, the
            set of variables we partially minimize over.
            Defaults to `np.zeros(p, np.bool)`.

        randomizer_scale : float
            Scale for IID components of randomizer.

        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']

        Returns
        -------

        L : `selection.randomized.convenience.threshold`

        """
        n, p = X.shape
        loglike = rr.glm.poisson(X, counts)

        # scale for randomizer seems kind of meaningless here...

        if active is None:
            active = np.zeros(p, np.bool)
        if candidate is None:
            candidate = ~active

        mean_diag = np.mean((X**2).sum(0))
        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(counts)

        return threshold(loglike, 
                         threshold_value,
                         candidate,
                         randomizer_scale, 
                         active=active,
                         randomizer=randomizer,
                         parametric_cov_estimator=parametric_cov_estimator)
