from itertools import product
import numpy as np

from scipy.stats import norm as ndist
from scipy.optimize import bisect

from regreg.affine import power_L

from .selective_MLE_utils import solve_barrier_nonneg

from ..distributions.api import discrete_family
from ..sampling.langevin import projected_langevin
from ..constraints.affine import sample_from_constraints

class query(object):

    def __init__(self, randomization, perturb=None):

        self.randomization = randomization
        self.perturb = perturb
        self._solved = False
        self._randomized = False
        self._setup = False

    # Methods reused by subclasses

    def randomize(self, perturb=None):

        if not self._randomized:
            self.randomized_loss, self._initial_omega = self.randomization.randomize(self.loss, self.epsilon, perturb=perturb)
        self._randomized = True

    def linear_decomposition(self, target_score_cov, target_cov, observed_target_state):
        """
        Compute out the linear decomposition
        of the score based on the target. This decomposition
        writes the (limiting CLT version) of the data in the score as linear in the
        target and in some independent Gaussian error.

        This second independent piece is conditioned on, resulting
        in a reconstruction of the score as an affine function of the target
        where the offset is the part related to this independent
        Gaussian error.
        """

        target_score_cov = np.atleast_2d(target_score_cov)
        target_cov = np.atleast_2d(target_cov)
        observed_target_state = np.atleast_1d(observed_target_state)

        linear_part = target_score_cov.T.dot(np.linalg.pinv(target_cov))
        offset = self.observed_score_state - linear_part.dot(observed_target_state) + score_offset

        return (linear_part, offset)

    def get_sampler(self):
        if hasattr(self, "_sampler"):
            return self._sampler

    def set_sampler(self, sampler):
        self._sampler = sampler

    sampler = property(get_sampler, set_sampler)

    # implemented by subclasses

    def solve(self):

        raise NotImplementedError('abstract method')

    def setup_sampler(self):
        """
        Setup query to prepare for sampling.
        Should set a few key attributes:

            - observed_score_state
            - num_opt_var
            - observed_opt_state
            - opt_transform

        """
        raise NotImplementedError('abstract method -- only keyword arguments')

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

    def solve(self):
        '''
        Ensure that each objective has been solved.
        '''
        for objective in self.objectives:
            if not objective._solved:
                objective.solve()

class optimization_sampler(object):

    def __init__(self):
        raise NotImplementedError("abstract method")

    def sample(self):
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
        using projected Langevin sampler with
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
        gradient : np.float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample = self.sample(*sample_args)

        if parameter is None:
            parameter = self.reference

        sample_test_stat = np.squeeze(np.array([test_stat(x) for x in sample]))

        target_inv_cov = np.linalg.inv(target_cov)
        delta = target_inv_cov.dot(parameter - self.reference)
        W = np.exp(sample.dot(delta))

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
                             level=0.9):
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
            sample = self.sample(*sample_args)
        else:
            ndraw = sample.shape[0]

        _intervals = optimization_intervals([(self, sample, target_cov, score_cov)],
                                            observed_target, ndraw)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            limits.append(_intervals.confidence_interval(keep, level=level))

        return np.array(limits)

    def coefficient_pvalues(self,
                            observed_target,
                            target_cov,
                            score_cov,
                            parameter=None,
                            sample_args=(),
                            sample=None,
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
            sample = self.sample(*sample_args)
        else:
            ndraw = sample.shape[0]

        if parameter is None:
            parameter = np.zeros(observed_target.shape[0])

        _intervals = optimization_intervals([(self, sample, target_cov, score_cov)],
                                            observed_target, ndraw)
        pvals = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            pvals.append(_intervals.pivot(keep, candidate=parameter[i], alternative=alternatives[i]))

        return np.array(pvals)

class langevin_sampler(optimization_sampler):

    '''
    Object to sample only optimization variables of a selective sampler
    fixing the observed score.
    '''

    def __init__(self,
                 observed_opt_state,
                 observed_score_state,
                 score_transform,
                 opt_transform,
                 projection,
                 grad_log_density,
                 log_density,
                 selection_info=None):

        '''
        Parameters
        ----------

        multi_view : `multiple_queries`
           Instance of `multiple_queries`. Attributes
           `objectives`, `score_info` are key
           attributed. (Should maybe change constructor
           to reflect only what is needed.)
        '''

        self.observed_opt_state = observed_opt_state.copy()
        self.observed_score_state = observed_score_state.copy()
        self.score_linear, self.score_offset = score_transform
        self.opt_linear, self.opt_offset = opt_transform
        self.projection = projection
        self.gradient = lambda opt: - grad_log_density(self.observed_score_state, opt)
        self.log_density = log_density
        self.selection_info = selection_info # a way to record what view and what was conditioned on -- not used in calculations

    def sample(self, ndraw, burnin, stepsize=None):
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

        stepsize : float
           Stepsize for Langevin sampler. Defaults
           to a crude estimate based on the
           dimension of the problem.

        Returns
        -------

        gradient : np.float
        '''

        if self.observed_opt_state.shape in ((), (0,)): # no opt variables to sample:
            return None

        if stepsize is None:
            stepsize = 1./max(len(self.observed_opt_state), 1)

        target_langevin = projected_langevin(self.observed_opt_state.copy(),
                                             self.gradient,
                                             self.projection,
                                             stepsize)

        samples = []

        for i in range(ndraw + burnin):
            target_langevin.next()
            if (i >= burnin):
                samples.append(target_langevin.state.copy())
        return np.asarray(samples)

    def crude_lipschitz(self):
        """
        A crude Lipschitz constant for the
        gradient of the log-density.
        Returns
        -------
        lipschitz : float

        """
        lipschitz = power_L(self.target_inv_cov)
        for transform, objective in zip(self.target_transform, self.objectives):
            lipschitz += power_L(transform[0])**2 * objective.randomization.lipschitz
            lipschitz += power_L(objective.score_transform[0])**2 * objective.randomization.lipschitz
        return lipschitz

class affine_gaussian_sampler(optimization_sampler):

    '''
    Sample from an affine truncated Gaussian
    '''

    def __init__(self,
                 affine_con,
                 initial_point,
                 observed_score_state,
                 log_density,
                 logdens_transform, # described how score enters log_density.
                 selection_info=None):

        '''
        Parameters
        ----------

        multi_view : `multiple_queries`
           Instance of `multiple_queries`. Attributes
           `objectives`, `score_info` are key
           attributed. (Should maybe change constructor
           to reflect only what is needed.)
        '''

        self.affine_con = affine_con
        self.initial_point = initial_point
        self.observed_score_state = observed_score_state
        self.selection_info = selection_info
        self.log_density = log_density
        self.logdens_transform = logdens_transform

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

        return sample_from_constraints(self.affine_con,
                                       self.initial_point,
                                       ndraw=ndraw,
                                       burnin=burnin)

    def selective_MLE(self, 
                      observed_target, 
                      cov_target, 
                      cov_target_score, 
                      feasible_point, 
                      solve_args={'tol':1.e-12}, 
                      alpha=0.1):
        """
        Selective MLE based on approximation of
        CGF.

        """
        prec_target = np.linalg.inv(cov_target)
        logdens_lin, logdens_off = self.logdens_transform
        target_lin = - logdens_lin.dot(cov_target_score.T.dot(prec_target)) # this determines how the conditional mean of optimization variables
                                                                            # vary with target
                                                                            # logdens_lin determines how the argument of the optimization density
                                                                            # depends on the score, not how the mean depends on score, hence the minus sign
        target_offset = self.affine_con.mean - target_lin.dot(observed_target)

        cov_opt = self.affine_con.covariance
        prec_opt = np.linalg.inv(cov_opt)

        conjugate_arg = prec_opt.dot(self.affine_con.mean)

        init_soln = feasible_point
        val, soln, hess = _solve_barrier_affine(conjugate_arg,
                                                prec_opt,
                                                self.affine_con,
                                                init_soln,
                                                **solve_args)

        final_estimator = observed_target + cov_target.dot(target_lin.T.dot(prec_opt.dot(self.affine_con.mean - soln)))
        ind_unbiased_estimator = observed_target + cov_target.dot(target_lin.T.dot(prec_opt.dot(self.affine_con.mean
                                                                                                - feasible_point)))
        L = target_lin.T.dot(prec_opt)
        observed_info_natural = prec_target + L.dot(target_lin) - L.dot(hess.dot(L.T))
        observed_info_mean = cov_target.dot(observed_info_natural.dot(cov_target))

        Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))
        pvalues = ndist.cdf(Z_scores)
        pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

        quantile = ndist.ppf(1 - alpha / 2.)
        intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                               final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

        return final_estimator, observed_info_mean, Z_scores, pvalues, intervals, ind_unbiased_estimator

    def reparam_map(self, theta, observed_target, cov_target, cov_target_score, feasible_point, solve_args={}):

        prec_target = np.linalg.inv(cov_target)
        ndim = prec_target.shape[0]
        logdens_lin, logdens_off = self.logdens_transform
        target_lin = - logdens_lin.dot(cov_target_score.T.dot(prec_target))
        target_offset = self.affine_con.mean - target_lin.dot(observed_target)

        cov_opt = self.affine_con.covariance
        prec_opt = np.linalg.inv(cov_opt)

        mean_param = target_lin.dot(theta)+target_offset
        conjugate_arg = prec_opt.dot(mean_param)
        init_soln = feasible_point
        val, soln, hess = solve_barrier_nonneg(conjugate_arg,
                                               prec_opt,
                                               init_soln,
                                               **solve_args)

        inter_map = cov_target.dot(target_lin.T.dot(prec_opt))
        param_map = theta + inter_map.dot(mean_param - soln)
        log_normalizer_map = (theta.T.dot(prec_target + target_lin.T.dot(prec_opt).dot(target_lin)).dot(theta))/2. \
                             - theta.T.dot(target_lin.T).prec_opt.dot(soln) - target_offset.T.dot(prec_opt).dot(target_offset)/2. \
                             + val - (param_map.T.dot(prec_target).param_map)/2.

        jacobian_map = (np.identity(ndim)+ inter_map.dot(target_lin)) - inter_map.dot(hess).dot(prec_opt.dot(target_lin))

        return param_map, log_normalizer_map, jacobian_map

class optimization_intervals(object):

    def __init__(self,
                 opt_sampling_info, # a sequence of (opt_sampler, opt_sample, target_cov, score_cov) objects
                                    # in theory all target_cov should be about the same...
                 observed,
                 nsample, # how large a normal sample
                 target_cov=None):

        # not all opt_samples will be of the same size as nsample 
        # let's repeat them as necessary
        
        tiled_sampling_info = []
        for opt_sampler, opt_sample, t_cov, score_cov in opt_sampling_info: 
            if opt_sample is not None:
                if opt_sample.shape[0] < nsample:
                    if opt_sample.ndim == 1:
                        tiled_opt_sample = np.tile(opt_sample, np.ceil(nsample / opt_sample.shape[0]))[:nsample]
                    else:
                        tiled_opt_sample = np.tile(opt_sample, (np.ceil(nsample / opt_sample.shape[0]), 1))[:nsample]
                else:
                    tiled_opt_sample = opt_sample[:nsample]
            else:
                tiled_sample = None
            tiled_sampling_info.append((opt_sampler, tiled_opt_sample, t_cov, score_cov))

        self.opt_sampling_info = tiled_sampling_info
        self._logden = 0
        for opt_sampler, opt_sample, _, _ in opt_sampling_info:
            self._logden += opt_sampler.log_density(opt_sampler.observed_score_state, opt_sample)

        self.observed = observed.copy() # this is our observed unpenalized estimator

        if target_cov is None:
            self.target_cov = 0
            for _, _, target_cov, _ in opt_sampling_info:
                self.target_cov += target_cov
            self.target_cov /= len(opt_sampling_info)

        self._normal_sample = np.random.multivariate_normal(mean=np.zeros(self.target_cov.shape[0]), 
                                                            cov=self.target_cov, 
                                                            size=(nsample,))

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
        for opt_sampler, opt_sample, _, score_cov in self.opt_sampling_info:
            cur_score_cov = linear_func.dot(score_cov)

            # cur_nuisance is in the view's score coordinates
            cur_nuisance = opt_sampler.observed_score_state - cur_score_cov * observed_stat / target_cov
            nuisance.append(cur_nuisance)
            translate_dirs.append(cur_score_cov / target_cov)

        weights = self._weights(sample_stat + candidate,  # normal sample under candidate
                                nuisance,                 # nuisance sufficient stats for each view
                                translate_dirs)               # points will be moved like sample * score_cov
        
        pivot = np.mean((sample_stat + candidate <= observed_stat) * weights) / np.mean(weights)

        if alternative == 'twosided':
            return 2 * min(pivot, 1 - pivot)
        elif alternative == 'less':
            return pivot
        else:
            return 1 - pivot

    def confidence_interval(self, linear_func, level=0.90, how_many_sd=20):

        sample_stat = self._normal_sample.dot(linear_func)
        observed_stat = self.observed.dot(linear_func)
        
        _norm = np.linalg.norm(linear_func)
        grid_min, grid_max = -how_many_sd * np.std(sample_stat), how_many_sd * np.std(sample_stat)

        def _rootU(gamma):
            return self.pivot(linear_func,
                              observed_stat + gamma,
                              alternative='less') - (1 - level) / 2.
        def _rootL(gamma):
            return self.pivot(linear_func,
                              observed_stat + gamma,
                              alternative='less') - (1 + level) / 2.

        upper = bisect(_rootU, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))
        lower = bisect(_rootL, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))

        return lower + observed_stat, upper + observed_stat

    # Private methods

    def _weights(self, 
                 sample_stat,
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
            score_sample = np.multiply.outer(sample_stat, translate_dirs[i]) + nuisance[i][None, :] # these are now score coordinates
            _lognum += opt_sampler.log_density(score_sample, opt_sample)

        _logratio = _lognum - self._logden
        _logratio -= _logratio.max()

        return np.exp(_logratio)

def naive_confidence_intervals(diag_cov, observed, alpha=0.1):
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

def _solve_barrier_affine(conjugate_arg,
                          precision,
                          constraints,
                          feasible_point=None,
                          step=1,
                          nstep=1000,
                          tol=1.e-8):

    con_linear = constraints.linear_part
    con_offset = constraints.offset
    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. \
                          + np.log(1.+ 1./((con_offset-con_linear.dot(u))/ scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) -con_linear.T.dot(1./(scaling + con_offset-con_linear.dot(u)) -
                                                                       1./(con_offset-con_linear.dot(u)))
    barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1./((scaling + con_offset-con_linear.dot(u))**2.)
                                                 + 1./((con_offset-con_linear.dot(u))**2.))).dot(con_linear)

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        newton_step = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * newton_step
            if np.all(con_offset-con_linear.dot(proposal) > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            proposal = current - step * newton_step
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + barrier_hessian(current))
    return current_value, current, hess


