
from __future__ import print_function
import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from .query import query, affine_gaussian_sampler

from .randomization import randomization
from ..base import restricted_estimator
from ..algorithms.debiased_lasso import (debiasing_matrix,
                                         pseudoinverse_debiasing_matrix)

#### High dimensional version
#### - parametric covariance
#### - Gaussian randomization

class group_lasso(query):


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
                  groups,
                  weights,
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

         self.ridge_term = ridge_term
         self.penalty = rr.group_lasso(groups,
                                       weights=weights,
                                       lagrange=1.)
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

        (self.initial_soln, 
         self.initial_subgrad) = self._solve_randomized_problem(
                                      perturb=perturb, 
                                      solve_args=solve_args)

        # assuming solution is non-zero here!

        active = []
        active_dirs = {}
        unpenalized = []
        overall = np.ones(p, np.bool)

        ordered_groups = []
        ordered_opt = []
        ordered_vars = []

        tol = 1.e-6

        for g in sorted(np.unique(self.penalty.groups)):
            group = self.penalty.groups == g

            soln = self.initial_soln
            if np.linalg.norm(soln[group]) * tol * np.linalg.norm(soln):
                ordered_groups.append(g)
                ordered_vars.extend(np.nonzero(group)[0])

                if self.penalty.weights[g] == 0:
                    unpenalized.append(g)
                    ordered_opt.append(soln[group])
                else:
                    active.append(g)
                    dir = soln[group] / np.linalg.norm(soln[group])
                    active_dirs[g] = dir
                    ordered_opt.append(np.linalg.norm(soln[group]) - self.penalty.weights[g])
            else:
                overall[group] = False

        self.selection_variable = {'directions': active_dirs,
                                   'active_groups':active}

        self._ordered_groups = ordered_groups

        # initial state for opt variables

        self.observed_opt_state = np.hstack(ordered_opt)

        _beta_unpenalized = restricted_estimator(self.loglike, 
                                                 overall, 
                                                 solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # form linear part

        num_opt_var = self.observed_opt_state.shape[0]

        # setup hessian and ridge term

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        opt_linear = np.dot(X.T, X[:, ordered_vars] * W[:, None])

        # set the observed score (data dependent) state

        # observed_score_state is
        # \nabla \ell(\bar{\beta}_E) + Q(\bar{\beta}_E) \bar{\beta}_E
        # in linear regression this is _ALWAYS_ -X^TY
        # 
        # should be asymptotically equivalent to
        # \nabla \ell(\beta^*) + Q(\beta^*)\beta^*

        self.observed_score_state = -opt_linear.dot(_beta_unpenalized)
        self.observed_score_state[~overall] += self.loglike.smooth_objective(beta_bar, 'grad')[~overall]

        # now adjust for ridge term

        for i, var in enumerate(ordered_vars):
            opt_linear[var, i] += self.ridge_term

        opt_offset = self.initial_subgrad
         
        # for group LASSO, we will have
        # a different sampler for each group
        # based on conditioning on all scalings
        # except that group

        # means targets will also need to say what group 
        # they are "targeting"...

        self._samplers = {}

        # dispersion will have to be adjusted for data splitting

        dispersion = 1.

        (prec_opt_linear,
         logdens_linear) = self._get_precision_opt_linear(opt_linear,
                                                          ordered_vars,
                                                          dispersion)

        for group, dens_info in _reference_density_info(soln, 
                                                        ordered_groups,
                                                        ordered_vars,
                                                        opt_linear,
                                                        opt_offset,
                                                        self.observed_score_state,
                                                        self.initial_subgrad,
                                                        self.penalty, 
                                                        prec_opt_linear).items():

            (dir_g,
             idx_g,
             implied_mean, 
             implied_variance, 
             log_det,
             log_cond_density) = dens_info
            
            group_idx = self.penalty.groups == group
            initial_scaling = np.linalg.norm(soln[group]) - self.penalty.weights[group]

            sampler = polynomial_gaussian_sampler(implied_mean,
                                                  implied_variance,
                                                  initial_scaling,
                                                  self.observed_score_state,
                                                  log_cond_density,
                                                  log_det,
                                                  (np.atleast_2d(logdens_linear.T[:,idx_g].dot(dir_g).T), 
                                                   opt_offset))
            self._samplers[group] = sampler

        self._setup = True

        return self.selection_variable

    def summary(self,
                observed_target, 
                group_assignments,
                target_cov, 
                target_score_cov, 
                alternatives,
                parameter=None,
                level=0.9,
                ndraw=10000,
                compute_intervals=False):

        # for smoke test, let's just 
        # use the first non-zero group for everything

        if parameter is None:
            parameter = np.zeros_like(observed_target)

        pvalues = np.zeros_like(observed_target)
        pivots = np.zeros_like(observed_target)
        intervals = np.zeros((parameter.shape[0], 2))

        for group in np.unique(group_assignments):
            group_idx = group_assignments == group

            (pvalues_, 
             pivots_,
             intervals_) = self._inference_for_target(
                               observed_target[group_idx],
                               group,
                               target_cov[group_idx][:, group_idx],
                               target_score_cov[group_idx],
                               [alternatives[i] for i in np.nonzero(group_idx)[0]],
                               parameter=parameter[group_idx],
                               level=level,
                               ndraw=ndraw,
                               compute_intervals=compute_intervals)
            pvalues[group_idx] = pvalues_
            pivots[group_idx] = pivots_
            intervals[group_idx] = np.array(intervals_)

        return pvalues, pivots, intervals

    def _inference_for_target(self,
                              observed_target, 
                              group,
                              target_cov, 
                              target_score_cov, 
                              alternatives,
                              opt_sample=None,
                              target_sample=None,
                              parameter=None,
                              level=0.9,
                              ndraw=10000,
                              compute_intervals=False):
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

        compute_intervals : bool
            Compute confidence intervals?

        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?
        """

        sampler = self._samplers[group]
        if parameter is None:
            parameter = np.zeros_like(observed_target)

        if opt_sample is None:
            opt_sample, logW = sampler.sample(ndraw)
        else:
            ndraw = opt_sample.shape[0]

        pivots = sampler.coefficient_pvalues(observed_target,
                                             target_cov,
                                             target_score_cov,
                                             parameter=parameter,
                                             sample=(opt_sample, logW),
                                             normal_sample=target_sample,
                                             alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = sampler.coefficient_pvalues(observed_target,
                                                  target_cov,
                                                  target_score_cov,
                                                  parameter=np.zeros_like(parameter),
                                                  sample=(opt_sample, logW),
                                                  normal_sample=target_sample,
                                                  alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:

            intervals = sampler.confidence_intervals(observed_target,
                                                     target_cov,
                                                     target_score_cov,
                                                     sample=(opt_sample, logW),
                                                     normal_sample=target_sample,
                                                     level=level)

        return pivots, pvalues, intervals


    def _get_precision_opt_linear(self, opt_linear, variables, dispersion=1):
        """
        Precision of randomization times columns of restricted Hessian
        """
        _, prec = self.randomizer.cov_prec 
        if np.asarray(prec).shape in [(), (0,)]:
            value = prec * opt_linear / dispersion
        else:
            value = prec.dot(opt_linear) / dispersion

        cond_precision = opt_linear.T.dot(value)
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(value.T) * dispersion # is this last dispersion correct?

        return value, logdens_linear

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

        initial_soln = problem.solve(quad, **solve_args) 
        initial_subgrad = -(self.loglike.smooth_objective(initial_soln, 
                                                          'grad') +
                            quad.objective(initial_soln, 'grad'))

        return initial_soln, initial_subgrad

    @staticmethod
    def gaussian(X,
                 Y,
                 groups,
                 weights,
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

        if sigma != 1:
            blah
            weights = copy(weights)
            for k in weights.keys():
                weights[k] = weights[k] / sigma**2

        return group_lasso(loglike, 
                           groups, 
                           weights,
                           ridge_term, 
                           randomizer)

    @staticmethod
    def logistic(X,
                 successes,
                 groups,
                 weights,
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

        return group_lasso(loglike, 
                           groups,
                           weights,
                           ridge_term, 
                           randomizer)

    @staticmethod
    def coxph(X,
              times,
              status,
              groups,
              weights,
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

        return group_lasso(loglike,
                           groups,
                           weights,
                           ridge_term,
                           randomizer)

    @staticmethod
    def poisson(X,
                counts,
                groups,
                weights,
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

        return group_lasso(loglike,
                           groups,
                           weights,
                           ridge_term,
                           randomizer)

    @staticmethod
    def sqrt_lasso(X,
                   Y,
                   groups,
                   weights,
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
        
        weights = copy(weights)
        for k in weights.keys():
            weights[k] = weights[k] * denom

        obj = lasso(loglike, 
                    groups,
                    weights,
                    ridge_term * denom,
                    randomizer,
                    perturb=perturb * denom)
        obj._sqrt_soln = soln

        return obj

# private functions

def _reference_density_info(soln, 
                            ordered_groups, # ordering is used in assumptions about columns opt_linear
                            ordered_variables,
                            opt_linear,
                            opt_offset,
                            observed_score_state,
                            observed_subgrad,
                            group_lasso_penalty, 
                            # randomization precision times opt_linear
                            precision_opt_linear, 
                            tol=1.e-6):
    '''
    
    Parameters
    ----------
    
    Compute generalized eigenvalues above and return
    function to evaluate jacobian as a function of $r_g=\|z_g\|_2$
    fixing everything in the optimization variables except $r_g$.
    
    Above, $A_0$ is the Hessian of loss evaluated at an appropriate point.
    '''
    pen = group_lasso_penalty       # shorthand 
    nz = soln != 0                  # nonzero
    nnz = nz.sum()                  # num nonzero
    Hr = np.zeros((nnz, nnz))       # restricted hessian   

    nz_groups = []

    for group in ordered_groups:
        group_idx = pen.groups == group
        group_soln = soln[pen.groups == group]

        ng = group_idx.sum()
        group_direction = u_g = group_soln / np.linalg.norm(group_soln)
        group_norm = r_g = np.linalg.norm(group_soln)   # really r_g^*
        group_weight = lambda_g = pen.weights[group]

        fraction = np.sqrt(r_g / (lambda_g + r_g))

        # one of the blocks in D_0^{1/2}
        group_block = (np.identity(ng) * fraction + (1 - fraction) * 
                       np.multiply.outer(u_g, u_g))
        group_P = np.identity(ng) - np.multiply.outer(u_g, u_g)
        nz_groups.append((group,         # a group index g
                          group_idx,   # indices where group==idx
                          group_block, 
                          group_P,
                          r_g,          
                          lambda_g,
                          group_direction)
                        )
            
    # setup the block hessian Hr=D_0^{1/2}A_0D_0^{1/2}
    
    ctr_g = 0
    for group_g in nz_groups:
        which_idx_g, sqrt_g = group_g[1], group_g[2]
        idx_g = slice(ctr_g, ctr_g + which_idx_g.sum())
        ctr_h = 0
        for group_h in nz_groups:
            which_idx_h, sqrt_h = group_h[1], group_h[2]
            idx_h = slice(ctr_h, ctr_h + which_idx_h.sum())
            H_hg = opt_linear[which_idx_h][:,idx_g] # E columns of Hessian + epsilon I
            assert(np.allclose(H_hg, opt_linear[which_idx_g][:,idx_h].T))
            if group_h[0] == group_g[0]: # subtract the identity matrix
                H_hg -= np.identity(H_hg.shape[0])
            Hr[idx_g][:,idx_h] += sqrt_h.dot(H_hg).dot(sqrt_g).T # multiply left and right by D_0^{1/2}
            ctr_h += which_idx_h.sum()
        ctr_g += which_idx_g.sum()
        
    implied_precision = np.dot(opt_linear.T,
                               precision_opt_linear)
    
    # compute (I+Hr)^{-1}Hr
    
    final_matrix = np.linalg.inv(Hr)
    
    ctr_g = 0
    ref_dens_info = {}

    for group_g in nz_groups:
        which_g, which_idx_g, _, P_g, r_g, lambda_g, u_g = group_g
        idx_g = slice(ctr_g, ctr_g + which_idx_g.sum())

        if which_idx_g.sum() > 1:
            block_g = final_matrix[idx_g][:,idx_g]
            block_g = P_g.dot(block_g).dot(P_g)
            # \tilde{\gamma}'s
            eigvals_g = np.linalg.eigvalsh(block_g)[1:]        

            # factors in the determinant
            factors_g = lambda_g / ((eigvals_g + 1) * r_g)           
            k_g = which_idx_g.sum()
            def logdet_g(factors_g, r_g, k_g, lambda_g, r):
                r = np.reshape(r, (-1))
                num = np.multiply.outer(factors_g, r - r_g)
                den = np.add.outer(lambda_g * np.ones_like(factors_g), 
                                     r)
                return np.squeeze(np.log(1 + num / den).sum(0) + 
                                  + np.log(lambda_g + r) * (k_g - 1))

            logdet_g = functools.partial(logdet_g, 
                                         factors_g, 
                                         r_g, 
                                         k_g, 
                                         lambda_g)
            
        else: 
            logdet_g = lambda r: np.zeros_like(r).reshape(-1)

        implied_variance = 1 / (np.dot(implied_precision[idx_g][:,idx_g],
                                       u_g) * u_g).sum()

        # zero out this group's coordinate in the solution

        soln_idx = soln.copy()[ordered_variables]
        soln_idx[idx_g] = 0
        offset = (observed_subgrad + 
                  opt_linear.dot(soln_idx))
        direction = precision_opt_linear[:,idx_g].dot(u_g)
        num_implied_mean = -((observed_score_state + offset) * 
                             direction).sum()
        implied_mean = num_implied_mean * implied_variance

        def log_cond_density(offset, 
                             direction, 
                             implied_variance, 
                             log_det,
                             r, 
                             score_state):
            r = np.reshape(r, (-1,))

            num_implied_mean = ((score_state + offset) * direction).sum()
            implied_mean = num_implied_mean * implied_variance

            return np.squeeze(
                    (log_det(r) - 0.5 * 
                    (r - implied_mean)**2 / implied_variance))

        log_cond_density_g = functools.partial(log_cond_density,
                                               offset,
                                               direction,
                                               implied_variance,
                                               logdet_g)

        ref_dens_info[which_g] = (u_g,
                                  idx_g,
                                  implied_mean,
                                  implied_variance,
                                  logdet_g,
                                  log_cond_density_g)

        ctr_g += which_idx_g.sum()
        
    return ref_dens_info

class polynomial_gaussian_sampler(affine_gaussian_sampler):

    """
    1-dimensional Gaussian density restricted to [0,\infty) times a polynomial
    """

    def __init__(self,
                 implied_mean,
                 implied_covariance,
                 initial_point,
                 observed_score_state,
                 log_cond_density,
                 log_det,
                 logdens_transform, # describes how score enters log_density.
                 selection_info=None,
                 useC=False):

        self.mean = implied_mean
        self.covariance = np.zeros((1, 1))
        self.covariance[0, 0] = implied_covariance

        self._log_cond_density = log_cond_density
        self._log_det = log_det
        self.initial_point = initial_point
        self.observed_score_state = observed_score_state
        self.selection_info = selection_info
        self.logdens_transform = logdens_transform

    def sample(self, ndraw):
        '''
        Sample from a Gaussian truncated at zero
        with our mean and covariance,
        but give weight based on `self.log_det`

        Parameters
        ----------

        ndraw : int
           How long a chain to return?

        '''

        mean, variance = self.mean, self.covariance[0,0]
        sd = np.sqrt(variance)
        Zscore = mean / sd
        selection_prob = ndist.sf(-Zscore)
        truncated_Z = ndist.ppf((1 - selection_prob) + selection_prob * np.random.sample(ndraw))
        truncated_normal = truncated_Z * sd + mean
        logW = self._log_det(truncated_normal)
        return truncated_normal.reshape((-1,1)), logW

    def selective_MLE(self, 
                      observed_target, 
                      target_cov, 
                      target_score_cov, 
                      # initial (observed) value of optimization variables -- 
                      # used as a feasible point.
                      # precise value used only for independent estimator 
                      init_soln, 
                      solve_args={'tol':1.e-12}, 
                      level=0.9):

        raise NotImplementedError

    def reparam_map(self, 
                    parameter_target, 
                    observed_target, 
                    target_cov, 
                    target_score_cov, 
                    init_soln, 
                    solve_args={'tol':1.e-12},
                    useC=True):

        raise NotImplementedError

    def _log_density_ray(self,
                         candidate,
                         direction,
                         nuisance,
                         gaussian_sample,
                         opt_sample):

        value = affine_gaussian_sampler._log_density_ray(self,
                                                         candidate,
                                                         direction,
                                                         nuisance,
                                                         gaussian_sample,
                                                         opt_sample)
        value += self._log_det(opt_sample)
        return value

# functions to construct targets of inference
# and covariance with score representation

def selected_targets(loglike, 
                     W, 
                     active_groups,
                     penalty,
                     sign_info={}, 
                     dispersion=None,
                     solve_args={'tol': 1.e-12, 'min_its': 50}):

    X, y = loglike.data
    n, p = X.shape
    features = []
    
    group_assignments = []
    for group in active_groups:
        group_idx = penalty.groups == group
        features.extend(np.nonzero(group_idx)[0])
        group_assignments.extend([group] * group_idx.sum())

    Xfeat = X[:, features]
    Qfeat = Xfeat.T.dot(W[:, None] * Xfeat)
    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    cov_target = np.linalg.inv(Qfeat)
    _score_linear = -Xfeat.T.dot(W[:, None] * X).T
    crosscov_target_score = _score_linear.dot(cov_target)
    alternatives = ['twosided'] * len(features)

    if dispersion is None:  # use Pearson's X^2
        dispersion = ((y - loglike.saturated_loss.mean_function(
            Xfeat.dot(observed_target))) ** 2 / W).sum() / (n - Xfeat.shape[1])

    return (observed_target, 
            group_assignments,
            cov_target * dispersion, 
            crosscov_target_score.T * dispersion, 
            alternatives)

def full_targets(loglike, 
                 W, 
                 active_groups,
                 penalty,
                 dispersion=None,
                 solve_args={'tol': 1.e-12, 'min_its': 50}):
    
    X, y = loglike.data
    n, p = X.shape
    features = []
    
    group_assignments = []
    for group in active_groups:
        group_idx = penalty.groups == group
        features.extend(np.nonzero(group_idx)[0])
        group_assignments.extend([group] * group_idx.sum())

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

    alternatives = ['twosided'] * len(features)

    return (observed_target, 
            group_assignments,
            cov_target * dispersion, 
            crosscov_target_score.T * dispersion, 
            alternatives)

def debiased_targets(loglike, 
                     W, 
                     active_groups,
                     penalty,
                     sign_info={}, 
                     dispersion=None,
                     approximate_inverse='JM',
                     debiasing_args={}):

    X, y = loglike.data
    n, p = X.shape
    features = []
    
    group_assignments = []
    for group in active_groups:
        group_idx = penalty.groups == group
        features.extend(np.nonzero(group_idx)[0])
        group_assignments.extend([group] * group_idx.sum())

    # relevant rows of approximate inverse

    if approximate_inverse == 'JM':
        Qinv_hat = np.atleast_2d(debiasing_matrix(X * np.sqrt(W)[:, None], 
                                                  features,
                                                  **debiasing_args)) / n
    else:
        Qinv_hat = np.atleast_2d(pseudoinverse_debiasing_matrix(X * np.sqrt(W)[:, None],
                                                                features,
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
                      (n - len(features)))

    alternatives = ['twosided'] * len(features)
    return (observed_target, 
            group_assignments,
            cov_target * dispersion, 
            crosscov_target_score.T * dispersion, 
            alternatives)

def form_targets(target, 
                 loglike, 
                 W, 
                 active_groups,
                 penalty,
                 **kwargs):
    _target = {'full':full_targets,
               'selected':selected_targets,
               'debiased':debiased_targets}[target]
    return _target(loglike,
                   W,
                   features,
                   penalty,
                   **kwargs)
