"""
Classes encapsulating some common workflows in randomized setting
"""

from copy import copy

import numpy as np
import regreg.api as rr

from .glm import (glm_greedy_step,
                  glm_threshold_score,
                  pairs_bootstrap_glm)
from .randomization import randomization
from .query import multiple_queries

from .lasso import lasso

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
