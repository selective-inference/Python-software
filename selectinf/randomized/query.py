import numpy as np
import pandas as pd
from scipy.stats import norm as ndist

from ..constraints.affine import constraints
from ..algorithms.barrier_affine import solve_barrier_affine_py

from .posterior_inference import posterior
from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from .approx_reference import approximate_grid_inference
from .exact_reference import exact_grid_inference

class query(object):
    r"""
    This class is the base of randomized selective inference
    based on convex programs.
    The main mechanism is to take an initial penalized program
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B)
    and add a randomization and small ridge term yielding
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B) -
        \langle \omega, B \rangle + \frac{\epsilon}{2} \|B\|^2_2
    """

    def __init__(self, randomization, perturb=None):

        """
        Parameters
        ----------
        randomization : `selection.randomized.randomization.randomization`
            Instance of a randomization scheme.
            Describes the law of $\omega$.
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """
        self.randomization = randomization
        self.perturb = perturb
        self._solved = False
        self._randomized = False
        self._setup = False

    # Methods reused by subclasses

    def randomize(self, perturb=None):

        """
        The actual randomization step.
        Parameters
        ----------
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """

        if not self._randomized:
            (self.randomized_loss,
             self._initial_omega) = self.randomization.randomize(self.loss,
                                                                 self.epsilon,
                                                                 perturb=perturb)
        self._randomized = True

    def get_sampler(self):
        if hasattr(self, "_sampler"):
            return self._sampler

    def set_sampler(self, sampler):
        self._sampler = sampler

    sampler = property(get_sampler, set_sampler, doc='Sampler of optimization (augmented) variables.')

    # implemented by subclasses

    def solve(self):

        raise NotImplementedError('abstract method')


class gaussian_query(query):

    """
    A class with Gaussian perturbation to the objective -- 
    easy to apply CLT to such things
    """

    def fit(self, perturb=None):

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

    # Private methods

    def _setup_sampler(self,
                       linear_part,
                       offset,
                       opt_linear,
                       observed_subgrad,
                       dispersion=1):

        A, b = linear_part, offset

        if not np.all(A.dot(self.observed_opt_state) - b <= 0):
            raise ValueError('constraints not satisfied')

        (cond_mean,
         cond_cov,
         cond_precision,
         regress_opt,
         M1,
         M2,
         M3) = self._setup_implied_gaussian(opt_linear,
                                            observed_subgrad,
                                            dispersion=dispersion)

        self.cond_mean, self.cond_cov = cond_mean, cond_cov

        affine_con = constraints(A,
                                 b,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        self.affine_con = affine_con
        self.opt_linear = opt_linear
        self.observed_subgrad = observed_subgrad

    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):

        cov_rand, prec = self.randomizer.cov_prec

        if np.asarray(prec).shape in [(), (0,)]:
            prod_score_prec_unnorm = self._unscaled_cov_score * prec
        else:
            prod_score_prec_unnorm = self._unscaled_cov_score.dot(prec)

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T) * prec
        else:
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T).dot(prec)

        # regress_opt is regression coefficient of opt onto score + u...

        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        M1 = prod_score_prec_unnorm * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T)

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

    def selective_MLE(self,
                      target_spec,
                      level=0.90,
                      solve_args={'tol': 1.e-12}):

        return selective_MLE(target_spec,
                             self.observed_opt_state,
                             self.affine_con.mean,
                             self.affine_con.covariance,
                             self.affine_con.linear_part,
                             self.affine_con.offset,
                             self.opt_linear,
                             self.M1,
                             self.M2,
                             self.M3,
                             self.observed_score_state + self.observed_subgrad,
                             solve_args=solve_args,
                             level=level,
                             useC=False)


    def posterior(self,
                  target_spec,
                  dispersion=1,
                  prior=None,
                  solve_args={'tol': 1.e-12}):
        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        prior : callable
            A callable object that takes a single argument
            `parameter` of the same shape as `observed_target`
            and returns (value of log prior, gradient of log prior)
        dispersion : float, optional
            Dispersion parameter for log-likelihood.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        if prior is None:
            Di = 1. / (200 * np.diag(target_spec.cov_target))

            def prior(target_parameter):
                grad_prior = -target_parameter * Di
                log_prior = -0.5 * np.sum(target_parameter ** 2 * Di)
                return log_prior, grad_prior

        return posterior(self,
                         target_spec,
                         dispersion,
                         prior,
                         solve_args=solve_args)

    def approximate_grid_inference(self,
                                   target_spec,
                                   useIP=True,
                                   solve_args={'tol': 1.e-12}):

        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        solve_args : dict, optional
            Arguments passed to solver.
        """

        G = approximate_grid_inference(self,
                                       target_spec,
                                       solve_args=solve_args,
                                       useIP=useIP)

        return G.summary(alternatives=target_spec.alternatives)

    def exact_grid_inference(self,
                             target_spec,
                             solve_args={'tol': 1.e-12}):

        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        solve_args : dict, optional
            Arguments passed to solver.
        """

        G = exact_grid_inference(self,
                                 target_spec,
                                 solve_args=solve_args)

        return G.summary(alternatives=target_spec.alternatives)


class multiple_queries(object):
    '''
    Combine several queries of a given data
    through randomized algorithms.
    '''

    def __init__(self, objectives):
        '''
        Parameters
        ----------
        objectives : sequence
           A sequences of randomized objective functions.
        Notes
        -----
        Each element of `objectives` must
        have a `setup_sampler` method that returns
        a description of the distribution of the
        data implicated in the objective function,
        typically through the score or gradient
        of the objective function.
        These descriptions are passed to a function
        `form_covariances` to linearly decompose
        each score in terms of a target
        and an asymptotically independent piece.
        Returns
        -------
        None
        '''

        self.objectives = objectives

    def fit(self):
        for objective in self.objectives:
            if not objective._setup:
                objective.fit()

    def summary(self,
                target_specs,
                # a sequence of target_specs
                # objects in theory all cov_target
                # should be about the same. as should the observed_target
                alternatives=None,
                parameter=None,
                level=0.9,
                ndraw=5000,
                burnin=2000,
                compute_intervals=False):

        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
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
        """

        observed_target = target_specs[0].observed_target
        alternatives = target_specs[0].alternatives
        
        if parameter is None:
            parameter = np.zeros_like(observed_target)

        if alternatives is None:
            alternatives = ['twosided'] * observed_target.shape[0]

        if len(self.objectives) != len(target_specs):
            raise ValueError("number of objectives and sampling cov infos do not match")

        self.opt_sampling_info = []
        for i in range(len(self.objectives)):
            if target_specs[i].cov_target is None or target_specs[i].regress_target_score is None:
                raise ValueError("did not input target and score covariance info")
            opt_sample, opt_logW = self.objectives[i].sampler.sample(ndraw, burnin)
            self.opt_sampling_info.append((self.objectives[i].sampler,
                                           opt_sample,
                                           opt_logW,
                                           target_specs[i].cov_target,
                                           target_specs[i].regress_target_score))

        pivots = self.coefficient_pvalues(observed_target,
                                          parameter=parameter,
                                          alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.coefficient_pvalues(observed_target,
                                               parameter=np.zeros_like(observed_target),
                                               alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.confidence_intervals(observed_target,
                                                  level)

        result = pd.DataFrame({'target': observed_target,
                               'pvalue': pvalues,
                               'lower_confidence': intervals[:, 0],
                               'upper_confidence': intervals[:, 1]})

        if not np.all(parameter == 0):
            result.insert(4, 'pivot', pivots)
            result.insert(5, 'parameter', parameter)

        return result

    def coefficient_pvalues(self,
                            observed_target,
                            parameter=None,
                            sample_args=(),
                            alternatives=None):

        '''
        Construct selective p-values
        for each parameter of the target.
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        parameter : ndarray (optional)
            A vector of parameters with shape `self.shape`
            at which to evaluate p-values. Defaults
            to `np.zeros(self.shape)`.
        sample_args : sequence
           Arguments to `self.sample` if sample is not found
           for a given objective.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        Returns
        -------
        pvalues : ndarray
        '''

        for i in range(len(self.objectives)):
            if self.opt_sampling_info[i][1] is None:
                _sample, _logW = self.objectives[i].sampler.sample(*sample_args)
                self.opt_sampling_info[i][1] = _sample
                self.opt_sampling_info[i][2] = _logW

        ndraw = self.opt_sampling_info[0][1].shape[0]  # nsample for normal samples taken from the 1st objective

        _intervals = optimization_intervals(self.opt_sampling_info,
                                            observed_target,
                                            ndraw)

        pvals = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            pvals.append(_intervals.pivot(keep, candidate=parameter[i], alternative=alternatives[i]))

        return np.array(pvals)

    def confidence_intervals(self,
                             target_specs,
                             sample_args=(),
                             level=0.9):

        '''
        Construct selective confidence intervals
        for each parameter of the target.
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        sample_args : sequence
           Arguments to `self.sample` if sample is not found
           for a given objective.
        level : float
            Confidence level.
        Returns
        -------
        limits : ndarray
            Confidence intervals for each target.
        '''

        for i in range(len(self.objectives)):
            if self.opt_sampling_info[i][1] is None:
                _sample, _logW = self.objectives[i].sampler.sample(*sample_args)
                self.opt_sampling_info[i][1] = _sample
                self.opt_sampling_info[i][2] = _logW

        ndraw = self.opt_sampling_info[0][1].shape[0]  # nsample for normal samples taken from the 1st objective

        _intervals = optimization_intervals(self.opt_sampling_info,
                                            observed_target,
                                            ndraw)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            limits.append(_intervals.confidence_interval(keep, level=level))

        return np.array(limits)


def naive_confidence_intervals(diag_cov, observed, level=0.9):
    """
    Compute naive Gaussian based confidence
    intervals for target.
    Parameters
    ----------
    diag_cov : diagonal of a covariance matrix
    observed : np.float
        A vector of observed data of shape `target.shape`
    alpha : float (optional)
        1 - confidence level.
    Returns
    -------
    intervals : np.float
        Gaussian based confidence intervals.
    """
    alpha = 1 - level
    diag_cov = np.asarray(diag_cov)
    p = diag_cov.shape[0]
    quantile = - ndist.ppf(alpha / 2)
    LU = np.zeros((2, p))
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        LU[0, j] = observed[j] - sigma * quantile
        LU[1, j] = observed[j] + sigma * quantile
    return LU.T


def naive_pvalues(diag_cov, observed, parameter):
    diag_cov = np.asarray(diag_cov)
    p = diag_cov.shape[0]
    pvalues = np.zeros(p)
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        pval = ndist.cdf((observed[j] - parameter[j]) / sigma)
        pvalues[j] = 2 * min(pval, 1 - pval)
    return pvalues

def selective_MLE(target_spec,
                  observed_soln,  # initial (observed) value of
                  # optimization variables -- used as a
                  # feasible point.  precise value used
                  # only for independent estimator
                  cond_mean,
                  cond_cov,
                  linear_part,
                  offset,
                  opt_linear,
                  M1,   
                  M2,
                  M3,
                  observed_score,
                  solve_args={'tol': 1.e-12},
                  level=0.9,
                  useC=False):

    """
    Selective MLE based on approximation of
    CGF.
    Parameters
    ----------
    observed_target : ndarray
        Observed estimate of target.
    cov_target : ndarray
        Estimated covaraince of target.
    regress_target_score : ndarray
        Estimated regression coefficient of target on score.
    observed_soln : ndarray
        Feasible point for optimization problem.
    cond_mean : ndarray
        Conditional mean of optimization variables given target.
    cond_cov : ndarray
        Conditional covariance of optimization variables given target.
    linear_part : ndarray
        Linear part of affine constraints: $\{o:Ao \leq b\}$
    offset : ndarray
        Offset part of affine constraints: $\{o:Ao \leq b\}$
    solve_args : dict, optional
        Arguments passed to solver.
    level : float, optional
        Confidence level.
    useC : bool, optional
        Use python or C solver.
    """

    (observed_target,
     cov_target,
     regress_target_score) = target_spec[:3]

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)

    prec_opt = np.linalg.inv(cond_cov)

    # this is specific to target
    
    T1 = regress_target_score.T.dot(prec_target)
    T2 = T1.T.dot(M2.dot(T1))
    T3 = T1.T.dot(M3.dot(T1)) 
    T4 = M1.dot(opt_linear).dot(cond_cov).dot(opt_linear.T.dot(M1.T.dot(T1)))
    T5 = T1.T.dot(M1.dot(opt_linear))

    prec_target_nosel = prec_target + T2 - T3

    _P = -(T1.T.dot(M1.dot(observed_score)) + T2.dot(observed_target)) ##flipped sign of second term here

    bias_target = cov_target.dot(T1.T.dot(-T4.dot(observed_target) + M1.dot(opt_linear.dot(cond_mean))) - _P)

    conjugate_arg = prec_opt.dot(cond_mean)

    if useC:
        solver = solve_barrier_affine_C
    else:
        solver = solve_barrier_affine_py

    val, soln, hess = solver(conjugate_arg,
                             prec_opt,
                             observed_soln,
                             linear_part,
                             offset,
                             **solve_args)

    final_estimator = cov_target.dot(prec_target_nosel).dot(observed_target) \
                      + regress_target_score.dot(M1.dot(opt_linear)).dot(cond_mean - soln) - bias_target

    observed_info_natural = prec_target_nosel + T3 - T5.dot(hess.dot(T5.T))

    unbiased_estimator = cov_target.dot(prec_target_nosel).dot(observed_target) - bias_target

    observed_info_mean = cov_target.dot(observed_info_natural.dot(cov_target))

    Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))

    pvalues = ndist.cdf(Z_scores)

    pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

    alpha = 1. - level

    quantile = ndist.ppf(1 - alpha / 2.)

    intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                           final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

    log_ref = val + conjugate_arg.T.dot(cond_cov).dot(conjugate_arg) / 2.

    result = pd.DataFrame({'MLE': final_estimator,
                           'SE': np.sqrt(np.diag(observed_info_mean)),
                           'Zvalue': Z_scores,
                           'pvalue': pvalues,
                           'lower_confidence': intervals[:, 0],
                           'upper_confidence': intervals[:, 1],
                           'unbiased': unbiased_estimator})

    return result, observed_info_mean, log_ref

