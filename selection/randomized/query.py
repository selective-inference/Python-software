import functools
from itertools import product

import numpy as np
from scipy.stats import norm as ndist
from scipy.optimize import bisect

from regreg.affine import power_L
import regreg.api as rr

from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C

from ..distributions.api import discrete_family
from ..constraints.affine import (sample_from_constraints,
                                  constraints)
from ..algorithms.softmax import softmax_objective

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
            self.randomized_loss, self._initial_omega = self.randomization.randomize(self.loss, self.epsilon, perturb=perturb)
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

    def setup_sampler(self):
        """
        Setup query to prepare for sampling.
        Should set a few key attributes:

            - observed_score_state
            - observed_opt_state
            - opt_transform

        """
        raise NotImplementedError('abstract method -- only keyword arguments')

    def summary(self,
                observed_target, 
                target_cov, 
                target_score_cov, 
                alternatives,
                opt_sample=None,
                target_sample=None,
                parameter=None,
                level=0.9,
                ndraw=10000,
                burnin=2000,
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

        burnin : int (optional)
            Defaults to 1000.

        compute_intervals : bool
            Compute confidence intervals?

        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?
        """

        if parameter is None:
            parameter = np.zeros_like(observed_target)

        if opt_sample is None:
            opt_sample, logW = self.sampler.sample(ndraw, burnin)
        else:
            if len(opt_sample) == 1: # only a sample, so weights are 1s
                opt_sample = opt_sample[0]
                logW = np.zeros(ndraw)
            else:
                opt_sample, logW = opt_sample
            ndraw = opt_sample.shape[0]

        pivots = self.sampler.coefficient_pvalues(observed_target,
                                                  target_cov,
                                                  target_score_cov,
                                                  parameter=parameter,
                                                  sample=(opt_sample, logW),
                                                  normal_sample=target_sample,
                                                  alternatives=alternatives)

        MLE_intervals = self.selective_MLE(observed_target,
                                           target_cov,
                                           target_score_cov)[5]

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target,
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

            MLE_intervals = self.selective_MLE(observed_target,
                                               target_cov,
                                               target_score_cov)[4]

            intervals = self.sampler.confidence_intervals(observed_target,
                                                          target_cov,
                                                          target_score_cov,
                                                          sample=(opt_sample, logW),
                                                          normal_sample=target_sample,
                                                          initial_guess=MLE_intervals,
                                                          level=level)

        return pivots, pvalues, intervals

    def selective_MLE(self,
                      observed_target, 
                      target_cov, 
                      target_score_cov, 
                      level=0.9,
                      solve_args={'tol':1.e-12}):
        """

        Parameters
        ----------

        observed_target : ndarray
            Observed estimate of target.

        target_cov : ndarray
            Estimated covaraince of target.

        target_score_cov : ndarray
            Estimated covariance of target and score of randomized query.

        level : float, optional
            Confidence level.

        solve_args : dict, optional
            Arguments passed to solver.

        """
        
        return self.sampler.selective_MLE(observed_target,
                                          target_cov,
                                          target_score_cov,
                                          self.observed_opt_state,
                                          level=level,
                                          solve_args=solve_args)


class gaussian_query(query):

    useC = True

    """
    A class with Gaussian perturbation to the objective -- 
    easy to apply CLT to such things
    """

    def fit(self, perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

    # Private methods

    def _setup_sampler(self, 
                       A_scaling,
                       b_scaling,
                       opt_linear,
                       opt_offset,
                       # optional dispersion parameter
                       # for covariance of randomization
                       dispersion=1):

        if not np.all(A_scaling.dot(self.observed_opt_state) - b_scaling <= 0):
            raise ValueError('constraints not satisfied')

        (cond_mean, 
         cond_cov, 
         cond_precision, 
         logdens_linear) = self._setup_implied_gaussian(opt_linear, 
                                                        opt_offset,
                                                        dispersion)

        def log_density(logdens_linear, offset, cond_prec, opt, score):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

        log_density = functools.partial(log_density, 
                                        logdens_linear, 
                                        opt_offset, 
                                        cond_precision)

        self.cond_mean, self.cond_cov = cond_mean, cond_cov

        affine_con = constraints(A_scaling,
                                 b_scaling,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        self.sampler = affine_gaussian_sampler(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               (logdens_linear, opt_offset),
                                               selection_info=self.selection_variable,
                                               useC=self.useC)

    def _setup_implied_gaussian(self, 
                                opt_linear, 
                                opt_offset,
                                # optional dispersion parameter
                                # for covariance of randomization
                                dispersion=1):

        _, prec = self.randomizer.cov_prec 
        prec = prec / dispersion

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
                observed_target,
                opt_sampling_info,  # a sequence of (target_cov, score_cov) 
                                    # objects in theory all target_cov
                                    # should be about the same...
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

        if parameter is None:
            parameter = np.zeros_like(observed_target)

        if alternatives is None:
            alternatives = ['twosided'] * observed_target.shape[0]

        if len(self.objectives) != len(opt_sampling_info):
            raise ValueError("number of objectives and sampling cov infos do not match")

        self.opt_sampling_info = []
        for i in range(len(self.objectives)):
            if opt_sampling_info[i][0] is None or opt_sampling_info[i][1] is None:
                raise ValueError("did not input target and score covariance info")
            opt_sample, opt_logW = self.objectives[i].sampler.sample(ndraw, burnin)
            self.opt_sampling_info.append((self.objectives[i].sampler, 
                                           opt_sample, 
                                           opt_logW,
                                           opt_sampling_info[i][0], 
                                           opt_sampling_info[i][1]))

        pivots = self.coefficient_pvalues(observed_target,
                                          parameter=parameter,
                                          alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.coefficient_pvalues(observed_target,
                                               parameter=parameter,
                                               alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.confidence_intervals(observed_target,
                                                  level)

        return pivots, pvalues, intervals
        

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

        ndraw = self.opt_sampling_info[0][1].shape[0] # nsample for normal samples taken from the 1st objective

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
                             observed_target,
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

        ndraw = self.opt_sampling_info[0][1].shape[0] # nsample for normal samples taken from the 1st objective

        _intervals = optimization_intervals(self.opt_sampling_info, 
                                            observed_target, 
                                            ndraw)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            limits.append(_intervals.confidence_interval(keep, level=level))

        return np.array(limits)       


class optimization_sampler(object):

    def __init__(self):
        raise NotImplementedError("abstract method")

    def sample(self):
        raise NotImplementedError("abstract method")

    def log_cond_density(self,
                         opt_sample,
                         target_sample,
                         transform=None):
        """
        Density of opt_sample | target_sample
        """
        raise NotImplementedError("abstract method")

    def hypothesis_test(self,
                        test_stat,
                        observed_value,
                        target_cov,
                        score_cov,
                        sample_args=(),
                        sample=None,
                        parameter=0,
                        alternative='twosided'):

        '''
        Sample `target` from selective density
        using sampler with
        gradient map `self.gradient` and
        projection map `self.projection`.

        Parameters
        ----------

        test_stat : callable
           Test statistic to evaluate on sample from
           selective distribution.

        observed_value : float
           Observed value of test statistic.
           Used in p-value calculation.

        sample_args : sequence
           Arguments to `self.sample` if sample is None.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc. If not None,
           `ndraw, burnin, stepsize` are ignored.

        parameter : np.float (optional)

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------

        pvalue : float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample, logW = self.sample(*sample_args)
            sample = np.atleast_2d(sample)

        if parameter is None:
            parameter = self.reference

        sample_test_stat = np.squeeze(np.array([test_stat(x) for x in sample]))

        target_inv_cov = np.linalg.inv(target_cov)
        delta = target_inv_cov.dot(parameter - self.reference)
        W = np.exp(sample.dot(delta) + logW)

        family = discrete_family(sample_test_stat, W)
        pval = family.cdf(0, observed_value)

        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        else:
            return 2 * min(pval, 1 - pval)

    def confidence_intervals(self,
                             observed_target,
                             target_cov,
                             score_cov,
                             sample_args=(),
                             sample=None,
                             normal_sample=None,
                             level=0.9,
                             initial_guess=None):
        '''

        Parameters
        ----------
 
        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.

        sample_args : sequence
           Arguments to `self.sample` if sample is None.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.

        level : float (optional)
            Specify the
            confidence level.

        initial_guess : np.float
            Initial guesses at upper and lower limits, optional.

        Notes
        -----

        Construct selective confidence intervals
        for each parameter of the target.

        Returns
        -------

        intervals : [(float, float)]
            List of confidence intervals.
        '''

        if sample is None:
            sample, logW = self.sample(*sample_args)
            sample = np.vstack([sample]*5) # why times 5?
            logW = np.hstack([logW]*5) 
        else:
            sample, logW = sample

        ndraw = sample.shape[0]

        _intervals = optimization_intervals([(self, 
                                              sample, 
                                              logW,
                                              target_cov, 
                                              score_cov)],
                                            observed_target, 
                                            ndraw, 
                                            normal_sample=normal_sample)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            if initial_guess is None:
                l, u = _intervals.confidence_interval(keep, level=level)
            else:
                l, u = _intervals.confidence_interval(keep, level=level,
                                                      guess=initial_guess[i])
            limits.append((l, u))

        return np.array(limits)

    def coefficient_pvalues(self,
                            observed_target,
                            target_cov,
                            score_cov,
                            parameter=None,
                            sample_args=(),
                            sample=None,
                            normal_sample=None,
                            alternatives=None):
        '''
        Construct selective p-values
        for each parameter of the target.

        Parameters
        ----------

        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.

        parameter : np.float (optional)
            A vector of parameters with shape `self.shape`
            at which to evaluate p-values. Defaults
            to `np.zeros(self.shape)`.

        sample_args : sequence
           Arguments to `self.sample` if sample is None.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.

        alternatives : list of ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------
        pvalues : np.float

        '''

        if alternatives is None:
            alternatives = ['twosided'] * observed_target.shape[0]

        if sample is None:
            sample, logW = self.sample(*sample_args)
        else:
            sample, logW = sample
            ndraw = sample.shape[0]

        if parameter is None:
            parameter = np.zeros(observed_target.shape[0])

        _intervals = optimization_intervals([(self, 
                                              sample, 
                                              logW,
                                              target_cov, 
                                              score_cov)],
                                            observed_target, 
                                            ndraw, 
                                            normal_sample=normal_sample)
        pvals = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            pvals.append(_intervals.pivot(keep, 
                                          candidate=parameter[i], 
                                          alternative=alternatives[i]))

        return np.array(pvals)

    def _reconstruct_score_from_target(self,
                                       target_sample,
                                       transform=None):
        if transform is not None:
            direction, nuisance = transform
            score_sample = (np.multiply.outer(target_sample,
                                              direction) + 
                            nuisance[None, :])
        else:
            score_sample = target_sample
        return score_sample

class affine_gaussian_sampler(optimization_sampler):

    '''
    Sample from an affine truncated Gaussian
    '''

    def __init__(self,
                 affine_con,
                 initial_point,
                 observed_score_state,
                 log_cond_density,
                 logdens_transform, # described how score enters log_density.
                 selection_info=None,
                 useC=False):

        '''
        Parameters
        ----------

        affine_con : `selection.constraints.affine.constraints`
             Affine constraints

        initial_point : ndarray
             Feasible point for affine constraints.

        observed_score_state : ndarray
             Observed score of convex loss (slightly modified).
             Essentially (asymptotically) equivalent 
             to $\nabla \ell(\beta^*) + 
             Q(\beta^*)\beta^*$ where $\beta^*$ is population
             minimizer. For linear regression, it is always
             $-X^Ty$.

        log_cond_density : callable
             Density of optimization variables given score

        logdens_transform : tuple
             Description of how conditional mean
             of optimization variables depends on score.

        selection_info : optional
             Function of optimization variables that
             will be conditioned on.

        useC : bool, optional
            Use python or C solver.
        
        '''

        self.affine_con = affine_con

        self.covariance = self.affine_con.covariance
        self.mean = self.affine_con.mean

        self.initial_point = initial_point
        self.observed_score_state = observed_score_state
        self.selection_info = selection_info
        self._log_cond_density = log_cond_density
        self.logdens_transform = logdens_transform
        self.useC = useC

    def log_cond_density(self,
                         opt_sample,
                         target_sample,
                         transform=None):

        if transform is not None:
            direction, nuisance = transform
            return self._log_density_ray(0,   # candidate
                                              # has been added to
                                              # target
                                         direction,
                                         nuisance,
                                         target_sample,
                                         opt_sample)
        else:
            # target must be in score coordinates
            score_sample = target_sample

            # probably should switch
            # order of signature
            return self._log_cond_density(opt_sample,
                                          score_sample)

    def sample(self, ndraw, burnin):
        '''
        Sample `target` from selective density
        using projected Langevin sampler with
        gradient map `self.gradient` and
        projection map `self.projection`.

        Parameters
        ----------

        ndraw : int
           How long a chain to return?

        burnin : int
           How many samples to discard?

        '''

        _sample = sample_from_constraints(self.affine_con,
                                          self.initial_point,
                                          ndraw=ndraw,
                                          burnin=burnin)
        return _sample, np.zeros(_sample.shape[0])

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
        """
        Selective MLE based on approximation of
        CGF.

        Parameters
        ----------

        observed_target : ndarray
            Observed estimate of target.

        target_cov : ndarray
            Estimated covaraince of target.

        target_score_cov : ndarray
            Estimated covariance of target and score of randomized query.

        init_soln : ndarray
            Feasible point for optimization problem.

        level : float, optional
            Confidence level.

        solve_args : dict, optional
            Arguments passed to solver.

        """

        return selective_MLE(observed_target, 
                             target_cov, 
                             target_score_cov, 
                             init_soln, 
                             self.mean,
                             self.covariance,
                             self.logdens_transform[0],
                             self.affine_con.linear_part,
                             self.affine_con.offset,
                             solve_args=solve_args,
                             level=level,
                             useC=self.useC)

    def reparam_map(self, 
                    parameter_target, 
                    observed_target, 
                    target_cov, 
                    target_score_cov, 
                    init_soln, 
                    solve_args={'tol':1.e-12},
                    useC=True):

        prec_target = np.linalg.inv(target_cov)
        ndim = prec_target.shape[0]
        logdens_lin, _ = self.logdens_transform
        target_lin = - logdens_lin.dot(target_score_cov.T.dot(prec_target))
        target_offset = self.mean - target_lin.dot(observed_target)

        cov_opt = self.covariance
        prec_opt = np.linalg.inv(cov_opt)

        mean_param = target_lin.dot(parameter_target) + target_offset
        conjugate_arg = prec_opt.dot(mean_param)

        if useC:
            solver = solve_barrier_affine_C
        else:
            solver = _solve_barrier_affine_py

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt, # JT: I think this quadratic is wrong should involve target_cov and target_lin too?
                                 init_soln,
                                 self.affine_con.linear_part,
                                 self.affine_con.offset,
                                 **solve_args)
            
        inter_map = target_cov.dot(target_lin.T.dot(prec_opt))
        param_map = parameter_target + inter_map.dot(mean_param - soln)
        log_normalizer_map = ((parameter_target.T.dot(prec_target + target_lin.T.dot(prec_opt).dot(target_lin)).dot(parameter_target))/2. 
                              - parameter_target.T.dot(target_lin.T).dot(prec_opt.dot(soln)) - target_offset.T.dot(prec_opt).dot(target_offset)/2. 
                              + val - (param_map.T.dot(prec_target).dot(param_map))/2.)

        jacobian_map = (np.identity(ndim) + inter_map.dot(target_lin)) - inter_map.dot(hess).dot(prec_opt.dot(target_lin))

        return param_map, log_normalizer_map, jacobian_map

    def _log_density_ray(self,
                         candidate,
                         direction,
                         nuisance,
                         gaussian_sample,
                         opt_sample):

        # implicitly caching (opt_sample, gaussian_sample) ?

        if (not hasattr(self, "_direction") or not 
            np.all(self._direction == direction)):

            logdens_lin, logdens_offset = self.logdens_transform

            if opt_sample.shape[1] == 1:

                prec = 1. / self.covariance[0, 0]
                quadratic_term = logdens_lin.dot(direction)**2 * prec
                arg = (logdens_lin.dot(nuisance + logdens_offset) + 
                       logdens_lin.dot(direction) * gaussian_sample +
                       opt_sample[:,0])
                linear_term = logdens_lin.dot(direction) * prec * arg
                constant_term = arg**2 * prec

                self._cache = {'linear_term':linear_term,
                               'quadratic_term':quadratic_term,
                               'constant_term':constant_term}
            else:
                self._direction = direction.copy()

                # density is a Gaussian evaluated at
                # O_i + A(N + (Z_i + theta) * gamma + b)

                # b is logdens_offset
                # A is logdens_linear
                # Z_i is gaussian_sample[i] (real-valued)
                # gamma is direction
                # O_i is opt_sample[i]

                # let arg1 = O_i
                # let arg2 = A(N+b + Z_i \cdot gamma)
                # then it is of the form (arg1 + arg2 + theta * A gamma)

                logdens_lin, logdens_offset = self.logdens_transform
                cov = self.covariance
                prec = np.linalg.inv(cov)
                linear_part = logdens_lin.dot(direction) # A gamma

                if 1 in opt_sample.shape:
                    pass # stop3 what's this for?
                cov = self.covariance

                quadratic_term = linear_part.T.dot(prec).dot(linear_part)

                arg1 = opt_sample.T
                arg2 = logdens_lin.dot(np.multiply.outer(direction, gaussian_sample) + 
                                       (nuisance + logdens_offset)[:,None])
                arg = arg1 + arg2
                linear_term = linear_part.T.dot(prec).dot(arg)
                constant_term = np.sum(prec.dot(arg) * arg, 0)

                self._cache = {'linear_term':linear_term,
                               'quadratic_term':quadratic_term,
                               'constant_term':constant_term}
        (linear_term, 
         quadratic_term,
         constant_term) = (self._cache['linear_term'], 
                           self._cache['quadratic_term'],
                           self._cache['constant_term'])
        return (-0.5 * candidate**2 * quadratic_term - 
                 candidate * linear_term - 0.5 * constant_term)

class optimization_intervals(object):

    def __init__(self,
                 opt_sampling_info, # a sequence of 
                                    # (opt_sampler, 
                                    #  opt_sample, 
                                    #  opt_logweights,
                                    #  target_cov, 
                                    #  score_cov) objects
                                    #  in theory all target_cov 
                                    #  should be about the same...
                 observed,
                 nsample, # how large a normal sample
                 target_cov=None,
                 normal_sample=None):

        # not all opt_samples will be of the same size as nsample 
        # let's repeat them as necessary
        
        tiled_sampling_info = []
        for (opt_sampler, 
             opt_sample, 
             opt_logW,
             t_cov, 
             t_score_cov) in opt_sampling_info: 
            if opt_sample is not None:
                if opt_sample.shape[0] < nsample:
                    if opt_sample.ndim == 1:
                        tiled_opt_sample = np.tile(opt_sample, 
                                              int(np.ceil(nsample / 
                                              opt_sample.shape[0])))[:nsample]
                        tiled_opt_logW = np.tile(opt_logW,
                                                 int(np.ceil(nsample / 
                                                             opt_logW.shape[0])))[:nsample]
                    else:
                        tiled_opt_sample = np.tile(opt_sample, 
                                              (int(np.ceil(nsample / 
                                              opt_sample.shape[0])), 1))[:nsample]
                        tiled_opt_logW = np.tile(opt_logW,
                                                 (int(np.ceil(nsample / 
                                                              opt_logW.shape[0])), 1))[:nsample]
                else:
                    tiled_opt_sample = opt_sample[:nsample]
                    tiled_opt_logW = opt_logW[:nsample]
            else:
                tiled_sample = None
            tiled_sampling_info.append((opt_sampler, 
                                        tiled_opt_sample, 
                                        tiled_opt_logW,
                                        t_cov, 
                                        t_score_cov))

        self.opt_sampling_info = tiled_sampling_info
        self._logden = 0
        for opt_sampler, opt_sample, opt_logW, _, _ in opt_sampling_info:

            self._logden += opt_sampler.log_cond_density(
                                opt_sample,
                                opt_sampler.observed_score_state,
                                transform=None) 
            self._logden -= opt_logW
            if opt_sample.shape[0] < nsample:
                self._logden = np.tile(self._logden, 
                                       int(np.ceil(nsample / 
                                       opt_sample.shape[0])))[:nsample]

        # this is our observed unpenalized estimator
        self.observed = observed.copy()

        # average covariances in case they might be different

        if target_cov is None:
            self.target_cov = 0
            for _, _, _, target_cov, _ in opt_sampling_info:
                self.target_cov += target_cov
            self.target_cov /= len(opt_sampling_info)

        if normal_sample is None:
            self._normal_sample = np.random.multivariate_normal(
                                      mean=np.zeros(self.target_cov.shape[0]), 
                                      cov=self.target_cov, 
                                      size=(nsample,))
        else:
            self._normal_sample = normal_sample

    def pivot(self,
              linear_func,
              candidate,
              alternative='twosided'):
        '''
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalue : np.float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        observed_stat = self.observed.dot(linear_func)
        sample_stat = self._normal_sample.dot(linear_func)

        target_cov = linear_func.dot(self.target_cov.dot(linear_func))

        nuisance = []
        translate_dirs = []

        for (opt_sampler, 
             opt_sample, 
             _, 
             _, 
             target_score_cov) in self.opt_sampling_info:

            cur_score_cov = linear_func.dot(target_score_cov)

            # cur_nuisance is in the view's score coordinates
            cur_nuisance = opt_sampler.observed_score_state - cur_score_cov * observed_stat / target_cov
            nuisance.append(cur_nuisance)
            translate_dirs.append(cur_score_cov / target_cov)

        weights = self._weights(sample_stat,  # normal sample 
                                candidate,    # candidate value
                                nuisance,       # nuisance sufficient stats for each view
                                translate_dirs) # points will be moved like sample * target_score_cov

        pivot = np.mean((sample_stat + candidate <= observed_stat) * weights) / np.mean(weights)

        if alternative == 'twosided':
            return 2 * min(pivot, 1 - pivot)
        elif alternative == 'less':
            return pivot
        else:
            return 1 - pivot

    def confidence_interval(self, 
                            linear_func, 
                            level=0.90, 
                            how_many_sd=20,
                            guess=None):

        sample_stat = self._normal_sample.dot(linear_func)
        observed_stat = self.observed.dot(linear_func)
        
        def _rootU(gamma):
            return self.pivot(linear_func,
                              observed_stat + gamma,
                              alternative='less') - (1 - level) / 2.
        def _rootL(gamma):
            return self.pivot(linear_func,
                              observed_stat + gamma,
                              alternative='less') - (1 + level) / 2.

        if guess is None:
            grid_min, grid_max = -how_many_sd * np.std(sample_stat), how_many_sd * np.std(sample_stat)
            upper = bisect(_rootU, grid_min, grid_max)
            lower = bisect(_rootL, grid_min, grid_max)
            
        else:
            delta = 0.5 * (guess[1] - guess[0])
            
            # find interval bracketing upper solution
            count = 0
            while True:
                Lu, Uu = guess[1] - delta, guess[1] + delta
                valU = _rootU(Uu)
                valL = _rootU(Lu)
                if valU * valL < 0:
                    break
                delta *= 2
                count += 1
            upper = bisect(_rootU, Lu, Uu)

            # find interval bracketing lower solution
            count = 0
            while True:
                Ll, Ul = guess[0] - delta, guess[0] + delta
                valU = _rootL(Ul)
                valL = _rootL(Ll)
                if valU * valL < 0:
                    break
                delta *= 2
                count += 1
            lower = bisect(_rootL, Ll, Ul)

        return lower + observed_stat, upper + observed_stat

    # Private methods

    def _weights(self, 
                 stat_sample,
                 candidate,
                 nuisance,
                 translate_dirs):

        # Here we should loop through the views
        # and move the score of each view 
        # for each projected (through linear_func) normal sample
        # using the linear decomposition

        # We need access to the map that takes observed_score for each view
        # and constructs the full randomization -- this is the reconstruction map
        # for each view

        # The data state for each view will be set to be N_i + A_i \hat{\theta}_i
        # where N_i is the nuisance sufficient stat for the i-th view's
        # data with respect to \hat{\theta} and N_i  will not change because
        # it depends on the observed \hat{\theta} and observed score of i-th view

        # In this function, \hat{\theta}_i will change with the Monte Carlo sample

        score_sample = []
        _lognum = 0
        for i, opt_info in enumerate(self.opt_sampling_info):
            opt_sampler, opt_sample = opt_info[:2]

            _lognum += opt_sampler.log_cond_density(opt_sample,
                                                    stat_sample + candidate,
                                                    transform=
                                                    (translate_dirs[i],
                                                     nuisance[i]))

        _logratio = _lognum - self._logden
        _logratio -= _logratio.max()

        return np.exp(_logratio)

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
    quantile = - ndist.ppf(alpha/2)
    LU = np.zeros((2, p))
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        LU[0,j] = observed[j] - sigma * quantile
        LU[1,j] = observed[j] + sigma * quantile
    return LU.T

def naive_pvalues(diag_cov, observed, parameter):
    diag_cov = np.asarray(diag_cov)
    p = diag_cov.shape[0]
    pvalues = np.zeros(p)
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        pval = ndist.cdf((observed[j] - parameter[j])/sigma)
        pvalues[j] = 2 * min(pval, 1-pval)
    return pvalues

# private function

def _solve_barrier_affine_py(conjugate_arg,
                             precision,
                             feasible_point,
                             con_linear,
                             con_offset,
                             step=1,
                             nstep=1000,
                             min_its=200,
                             tol=1.e-10):

    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. \
                          + np.log(1.+ 1./((con_offset - con_linear.dot(u))/ scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) - con_linear.T.dot(1./(scaling + con_offset - con_linear.dot(u)) -
                                                                       1./(con_offset - con_linear.dot(u)))
    barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1./((scaling + con_offset-con_linear.dot(u))**2.)
                                                 + 1./((con_offset-con_linear.dot(u))**2.))).dot(con_linear)

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(con_offset-con_linear.dot(proposal) > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + barrier_hessian(current))
    return current_value, current, hess

def _solve_barrier_nonneg(conjugate_arg,
                          precision,
                          feasible_point=None,
                          step=1,
                          nstep=1000,
                          tol=1.e-8):

    scaling = np.sqrt(np.diag(precision))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. + np.log(1.+ 1./(u / scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) + (1./(scaling + u) - 1./u)
    barrier_hessian = lambda u: (-1./((scaling + u)**2.) + 1./(u**2.))

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(proposal > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + np.diag(barrier_hessian(current)))
    return current_value, current, hess

def selective_MLE(observed_target, 
                  target_cov, 
                  target_score_cov, 
                  init_soln, # initial (observed) value of optimization variables -- 
                             # used as a feasible point.
                             # precise value used only for independent estimator 
                  cond_mean,
                  cond_cov,
                  logdens_linear,
                  linear_part,
                  offset,
                  solve_args={'tol':1.e-12}, 
                  level=0.9,
                  useC=False):
    """
    Selective MLE based on approximation of
    CGF.

    Parameters
    ----------

    observed_target : ndarray
        Observed estimate of target.

    target_cov : ndarray
        Estimated covaraince of target.
       
    target_score_cov : ndarray
        Estimated covariance of target and score of randomized query.
    
    init_soln : ndarray
        Feasible point for optimization problem.

    cond_mean : ndarray
        Conditional mean of optimization variables given target.

    cond_cov : ndarray
        Conditional covariance of optimization variables given target.
    
    logdens_linear : ndarray
        Describes how conditional mean of optimization
        variables varies with target.
    
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

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(target_cov)

    # target_lin determines how the conditional mean of optimization variables
    # vary with target
    # logdens_linear determines how the argument of the optimization density
    # depends on the score, not how the mean depends on score, hence the minus sign

    target_lin = - logdens_linear.dot(target_score_cov.T.dot(prec_target)) 
    target_offset = cond_mean - target_lin.dot(observed_target)

    prec_opt = np.linalg.inv(cond_cov)

    conjugate_arg = prec_opt.dot(cond_mean)

    if useC:
        solver = solve_barrier_affine_C
    else:
        solver = solve_barrier_affine_py

    val, soln, hess = solver(conjugate_arg,
                             prec_opt,
                             init_soln,
                             linear_part,
                             offset,
                             **solve_args)

    final_estimator = observed_target + target_cov.dot(target_lin.T.dot(prec_opt.dot(cond_mean - soln)))
    ind_unbiased_estimator = observed_target + target_cov.dot(target_lin.T.dot(prec_opt.dot(cond_mean
                                                                                            - init_soln)))
    L = target_lin.T.dot(prec_opt)
    observed_info_natural = prec_target + L.dot(target_lin) - L.dot(hess.dot(L.T))
    observed_info_mean = target_cov.dot(observed_info_natural.dot(target_cov))

    Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))
    pvalues = ndist.cdf(Z_scores)
    pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

    alpha = 1 - level
    quantile = ndist.ppf(1 - alpha / 2.)
    intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                           final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

    return final_estimator, observed_info_mean, Z_scores, pvalues, intervals, ind_unbiased_estimator


def normalizing_constant(target_parameter,
                         observed_target,
                         target_cov,
                         target_score_cov,
                         feasible_point,
                         cond_mean,
                         cond_cov,
                         logdens_linear,
                         linear_part,
                         offset,
                         useC=False):

    """

    Approximation of normalizing constant
    in affine constrained Gaussian.

    Parameters
    ----------

    observed_target : ndarray
        Observed estimate of target.

    target_cov : ndarray
        Estimated covaraince of target.
       
    target_score_cov : ndarray
        Estimated covariance of target and score of randomized query.
    
    init_soln : ndarray
        Feasible point for optimization problem.

    cond_mean : ndarray
        Conditional mean of optimization variables given target.

    cond_cov : ndarray
        Conditional covariance of optimization variables given target.
    
    logdens_linear : ndarray
        Describes how conditional mean of optimization
        variables varies with target.
    
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

    target_parameter = np.atleast_1d(target_parameter)

    cond_precision = np.linalg.inv(cond_cov)
    prec_target = np.linalg.inv(target_cov)
    target_linear = -logdens_linear.dot(target_score_cov.dot(prec_target))
    nuisance_correction = target_linear.dot(observed_target)
    corrected_mean = cond_mean - nuisance_correction

    # rest of the objective is the target mahalanobis distance
    # plus the mahalanobis distance for optimization variables
    # this includes a term linear in the target, i.e.
    # the source of `target_linear`

    ntarget = target_cov.shape[0]
    nopt = cond_cov.shape[0]
    full_Q = np.zeros((ntarget + nopt,
                       ntarget + nopt))
    full_Q[:ntarget][:,:ntarget] = (prec_target + target_linear.T.dot(cond_precision.dot(target_linear)))
    full_Q[:ntarget][:,ntarget:] = -target_linear.dot(cond_precision)
    full_Q[ntarget:][:,:ntarget] = (-target_linear.dot(cond_precision)).T
    full_Q[ntarget:][:,ntarget:] = cond_precision

    linear_term = np.hstack([-prec_target.dot(target_parameter) + 
                              corrected_mean.dot(cond_precision).dot(target_linear), 
                              -cond_precision.dot(corrected_mean)])

    constant_term = 0.5 * (np.sum(target_parameter * prec_target.dot(target_parameter)) +
                           np.sum(corrected_mean * cond_precision.dot(corrected_mean)))

    full_con_linear = np.zeros((linear_part.shape[0],
                                ntarget + nopt))
    full_con_linear[:,ntarget:] = linear_part
    full_feasible = np.zeros(ntarget + nopt)
    full_feasible[ntarget:] = feasible_point

    solve_args={'tol':1.e-12}

    if useC:
        solver = solve_barrier_affine_C
    else:
        solver = solve_barrier_affine_py

    value, soln, hess = solver(-linear_term,
                                full_Q,
                                full_feasible,
                                full_con_linear,
                                offset,
                                **solve_args)
    return (-value + 0.5 * np.sum(target_parameter * prec_target.dot(target_parameter)), 
             soln[:ntarget], 
             hess[:ntarget][:,:ntarget])

