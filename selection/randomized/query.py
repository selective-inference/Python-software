from itertools import product
import numpy as np

from scipy.stats import norm as ndist
from scipy.optimize import bisect

from regreg.affine import power_L

from ..distributions.api import discrete_family, intervals_from_sample
from ..sampling.langevin import projected_langevin
from .target import (targeted_sampler,
                     bootstrapped_target_sampler)
from .reconstruction import (reconstruct_opt,
                             reconstruct_full_from_internal)


class query(object):

    def __init__(self, randomization):

        self.randomization = randomization
        self._solved = False
        self._randomized = False
        self._setup = False

    # Methods reused by subclasses

    def randomize(self):

        if not self._randomized:
            self.randomized_loss, self._initial_omega = self.randomization.randomize(self.loss, self.epsilon)
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

        offset = self.observed_score_state - linear_part.dot(observed_target_state)

        # now compute the composition of this map with
        # self.score_transform

        score_linear, score_offset = self.score_transform
        composition_linear_part = score_linear.dot(linear_part)

        composition_offset = score_linear.dot(offset) + score_offset

        return (composition_linear_part, composition_offset)

    # the default log conditional density of state given data 
    # with no conditioning or marginalizing

    def log_density(self, internal_state, opt_state):
        full_state = reconstruct_full_from_internal(self, internal_state, opt_state)
        return self.randomization.log_density(full_state)

    def grad_log_density(self, internal_state, opt_state):
        full_state = reconstruct_full_from_internal(self, internal_state, opt_state)
        return self.randomization.gradient(full_state)

     # implemented by subclasses

    def grad_log_jacobian(self, opt_state):
        """
        log_jacobian depends only on data through
        Hessian at \bar{\beta}_E which we
        assume is close to Hessian at \bar{\beta}_E^*
        """
        # needs to be implemented for group lasso
        return self.derivative_logdet_jacobian(opt_state[self.scaling_slice])

    def jacobian(self, opt_state):
        """
        log_jacobian depends only on data through
        Hessian at \bar{\beta}_E which we
        assume is close to Hessian at \bar{\beta}_E^*
        """
        # needs to be implemented for group lasso
        return 1.

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
            - score_transform

        """
        raise NotImplementedError('abstract method -- only keyword arguments')

    def projection(self, opt_state):

        raise NotImplementedError('abstract method -- projection of optimization variables')

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

    def setup_sampler(self, form_covariances):
        '''
        Parameters
        ----------
        form_covariances : callable
           A callable used to decompose
           target of inference and the score
           of each objective.
        Notes
        -----
        This function sets the initial
        `opt_state` of all optimization
        variables in each view.
        We also store a reference to `form_covariances`
        which is called in the
        construction of `targeted_sampler`.
        Returns
        -------
        None
        '''

        self.form_covariances = form_covariances

        nqueries = self.nqueries = len(self.objectives)

        self.score_info = []
        self.nboot = []
        for objective in self.objectives:
            score_ = objective.setup_sampler()
            self.score_info.append(score_)
            self.nboot.append(objective.nboot)

        curr_randomization_length = 0
        self.randomization_slice = []
        for objective in self.objectives:
            randomization_length = objective.randomization.shape[0]
            self.randomization_slice.append(slice(curr_randomization_length,
                                                  curr_randomization_length + randomization_length))
            curr_randomization_length = curr_randomization_length + randomization_length
        self.total_randomization_length = curr_randomization_length

    def setup_opt_state(self):
        self.num_opt_var = 0
        self.opt_slice = []

        for objective in self.objectives:
            self.opt_slice.append(slice(self.num_opt_var, self.num_opt_var + objective.num_opt_var))
            self.num_opt_var += objective.num_opt_var
        self.observed_opt_state = np.zeros(self.num_opt_var)
        for i in range(len(self.objectives)):
            if self.objectives[i].num_opt_var > 0:
                self.observed_opt_state[self.opt_slice[i]] = self.objectives[i].observed_opt_state

    def setup_target(self,
                     target_info,
                     observed_target_state,
                     reference=None,
                     target_set=None,
                     parametric=False):
        '''
        Parameters
        ----------
        target_info : object
           Passed as first argument to `self.form_covariances`.

        observed_target_state : np.float
           Observed value of the target estimator.

        reference : np.float (optional)
           Reference parameter for Gaussian approximation
           of target.

        target_set : sequence (optional)
           Which coordinates of target are really
           of interest. If not None, then coordinates
           not in target_set are assumed to have 0
           mean in the sampler.

        Notes
        -----

        The variable `target_set` can be used for
        a selected model test when some functionals
        are assumed to have 0 mean in the limiting
        Gaussian approximation. This can
        sometimes mean an increase in power.

        Returns
        -------

        target : targeted_sampler
            An instance of `targeted_sampler` that
            can be used to sample, test hypotheses,
            form intervals.
        '''

        self.setup_opt_state()

        return targeted_sampler(self,
                                target_info,
                                observed_target_state,
                                self.form_covariances,
                                target_set=target_set,
                                reference=reference,
                                parametric=parametric)

    def setup_bootstrapped_target(self,
                                  target_bootstrap,
                                  observed_target_state,
                                  target_alpha,
                                  target_set=None,
                                  reference=None,
                                  boot_size=None):

        self.setup_opt_state()

        return bootstrapped_target_sampler(self,
                                           target_bootstrap,
                                           observed_target_state,
                                           target_alpha,
                                           target_set=target_set,
                                           reference=reference,
                                           boot_size=boot_size)


class optimization_sampler(object):

    '''
    Object to sample only optimization variables of a selective sampler
    fixing the observed score.
    '''

    def __init__(self,
                 multi_view):

        '''
        Parameters
        ----------

        multi_view : `multiple_queries`
           Instance of `multiple_queries`. Attributes
           `objectives`, `score_info` are key
           attributed. (Should maybe change constructor
           to reflect only what is needed.)
        '''

        # sampler will draw samples for bootstrap
        # these are arguments to target_info and score_bootstrap
        # nonparamteric bootstrap is np.random.choice(n, size=(n,), replace=True)
        # residual bootstrap might be X_E.dot(\bar{\beta}_E)
        # + np.random.choice(resid, size=(n,), replace=True)

        # if target_set is not None, we assume that
        # these coordinates (specified by a list of coordinates) of target
        # is assumed to be independent of the rest
        # the corresponding block of `target_cov` is zeroed out

        # make sure we setup the queries

        multi_view.setup_sampler(form_covariances=None)
        multi_view.setup_opt_state()

        # we need these attributes of multi_view
        self.multi_view = multi_view

        self.nqueries = len(multi_view.objectives)
        self.opt_slice = multi_view.opt_slice
        self.objectives = multi_view.objectives
        self.nboot = multi_view.nboot

        self.total_randomization_length = multi_view.total_randomization_length
        self.randomization_slice = multi_view.randomization_slice

        # set the observed state

        self.observed_state = np.zeros_like(multi_view.observed_opt_state)
        self.observed_state[:] = multi_view.observed_opt_state

        # added for the reconstruction map in case we marginalize over optimization variables

        randomization_length_total = 0
        self.randomization_slice = []
        for i in range(self.nqueries):
            self.randomization_slice.append(
                slice(randomization_length_total, randomization_length_total + self.objectives[i].ndim))
            randomization_length_total += self.objectives[i].ndim

        self.randomization_length_total = randomization_length_total

        # We implicitly assume that we are sampling a target
        # independent of the data in each view

        self.observed_score = [] # in the view's coordinates
        self.score_info = []
        for i in range(self.nqueries):
            obj = self.objectives[i]
            score_linear, score_offset = obj.score_transform
            self.observed_score.append(obj.observed_score_state)
            self.score_info.append(obj.score_transform)

    def projection(self, state):
        '''
        Projection map of projected Langevin sampler.
        Parameters
        ----------
        state : np.float
           State of sampler made up of `(target, opt_vars)`.
           Typically, the projection will only act on
           `opt_vars`.
        Returns
        -------
        projected_state : np.float
        '''

        opt_state = state
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nqueries):
            new_opt_state[self.opt_slice[i]] = self.objectives[i].projection(opt_state[self.opt_slice[i]])
        return new_opt_state

    def gradient(self, state):
        """
        Gradient only w.r.t. opt variables
        """

        opt_state = state
        opt_grad = np.zeros_like(opt_state)

        # randomization_gradient are gradients of a CONVEX function

        for i in range(self.nqueries):
            opt_linear, opt_offset = self.objectives[i].opt_transform
            opt_grad[self.opt_slice[i]] = \
                opt_linear.T.dot(self.objectives[i].grad_log_density(self.observed_score[i], opt_state[self.opt_slice[i]]))
        return -opt_grad

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
        keep_opt : bool
           Should we return optimization variables
           as well as the target?
        Returns
        -------
        gradient : np.float
        '''

        if stepsize is None:
            stepsize = 1./len(self.observed_state) 

        target_langevin = projected_langevin(self.observed_state.copy(),
                                             self.gradient,
                                             self.projection,
                                             stepsize)

        samples = []

        for i in range(ndraw + burnin):
            target_langevin.next()
            if (i >= burnin):
                samples.append(target_langevin.state.copy())
        return np.asarray(samples)

    def setup_target(self, 
                     target_info, 
                     form_covariances, 
                     parametric=False):
        """
        This computes the matrices used in the linear decomposition
        that will be used in computing weights for the sampler.
        """

        self.score_cov = []
        self.log_densities = []

        target_cov_sum = 0

        # we should pararallelize this over all views at once ?
        for i in range(self.nqueries):
            view = self.objectives[i]
            self.log_densities.append(view.log_density)
            score_info = view.setup_sampler(form_covariances)
            if parametric == False:
                target_cov, cross_cov = form_covariances(target_info,  
                                                         cross_terms=[score_info],
                                                         nsample=self.nboot[i])
            else:
                target_cov, cross_cov = form_covariances(target_info, 
                                                         cross_terms=[score_info])

            target_cov_sum += target_cov
            self.score_cov.append(cross_cov)

        self.target_cov = target_cov_sum / self.nqueries
        self.target_invcov = np.linalg.inv(self.target_cov)

    def hypothesis_test(self,
                        test_stat,
                        observed_value,
                        ndraw=10000,
                        burnin=2000,
                        stepsize=None,
                        sample=None,
                        parameter=None,
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
        ndraw : int
           How long a chain to return?
        burnin : int
           How many samples to discard?
        stepsize : float
           Stepsize for Langevin sampler. Defaults
           to a crude estimate based on the
           dimension of the problem.
        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc. If not None,
           `ndraw, burnin, stepsize` are ignored.
        parameter : np.float (optional)
           If not None, defaults to `self.reference`.
           Otherwise, sample is reweighted using Gaussian tilting.
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        gradient : np.float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample = self.sample(ndraw, burnin, stepsize=stepsize)

        if parameter is None:
            parameter = self.reference

        sample_test_stat = np.squeeze(np.array([test_stat(x) for x in sample]))

        delta = self.target_inv_cov.dot(parameter - self.reference)
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
                             ndraw=10000,
                             burnin=2000,
                             stepsize=None,
                             sample=None,
                             level=0.9):
        '''
        Parameters
        ----------
        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.
        ndraw : int
           How long a chain to return?
        burnin : int
           How many samples to discard?
        stepsize : float
           Stepsize for Langevin sampler. Defaults
           to a crude estimate based on the
           dimension of the problem.
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
            sample = self.sample(ndraw, burnin, stepsize=stepsize)

        _intervals = optimization_intervals(self,
                                            sample,
                                            observed_target)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            limits.append(_intervals.confidence_interval(keep, level=level))

        return np.array(limits)

    def coefficient_pvalues(self,
                            observed_target,
                            parameter=None,
                            ndraw=10000,
                            burnin=2000,
                            stepsize=None,
                            sample=None,
                            alternative='twosided'):
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
        ndraw : int
           How long a chain to return?
        burnin : int
           How many samples to discard?
        stepsize : float
           Stepsize for Langevin sampler. Defaults
           to a crude estimate based on the
           dimension of the problem.
        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalues : np.float

        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample = self.sample(ndraw, burnin, stepsize=stepsize)

        if parameter is None:
            parameter = np.zeros(observed_target.shape[0])

        _intervals = optimization_intervals(self,
                                            sample,
                                            observed_target)
        pvals = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            pvals.append(_intervals.pivot(keep, candidate=parameter[i], alternative=alternative))

        return np.array(pvals)

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

    def log_density(self, internal_state, opt_state):
        '''
        Log of randomization density at current state.
        Parameters
        ----------
        internal_state : sequence
           Sequence of internal scores for each view (i.e.
           in their own coordinate systems).

        Returns
        -------
        density : np.float
            Has number of rows as `opt_state` if 2-dimensional.
        '''

        value = np.zeros(opt_state.shape[0])

        for i in range(self.nqueries):
            log_dens = self.objectives[i].log_density
            value += log_dens(internal_state[i], opt_state[:, self.opt_slice[i]]) # may have to broadcast shape here
        return np.squeeze(value)

class optimization_intervals(object):

    def __init__(self,
                 opt_sampler,
                 opt_sample,
                 observed):

        self._logden = opt_sampler.log_density(opt_sampler.observed_score, opt_sample)

        self.observed = observed.copy() # this is our observed unpenalized estimator

        # setup_target has been called on opt_sampler
        self.opt_sampler = opt_sampler
        self.opt_sample = opt_sample

        self.target_cov = opt_sampler.target_cov
        self._normal_sample = np.random.multivariate_normal(mean=np.zeros(self.target_cov.shape[0]), 
                                                            cov=self.target_cov, 
                                                            size=(opt_sample.shape[0],))

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
        score_cov = []
        for i in range(len(self.opt_sampler.objectives)):
            cur_score_cov = linear_func.dot(self.opt_sampler.score_cov[i])

            cur_nuisance = self.opt_sampler.observed_score[i] - cur_score_cov * observed_stat / target_cov

            # cur_nuisance is in the view's internal coordinates
            score_linear, score_offset = self.opt_sampler.score_info[i]
            # final_nuisance is on the scale of the original randomization
            final_nuisance = score_linear.dot(cur_nuisance) + score_offset
            nuisance.append(cur_nuisance)

            score_cov.append(cur_score_cov / target_cov)


        weights = self._weights(sample_stat + candidate,  # normal sample under candidate
                                nuisance,                 # nuisance sufficient stats for each view
                                score_cov,                # points will be moved like sample * score_cov
                                self.opt_sampler.log_densities)
        
        pivot = np.mean((sample_stat <= observed_stat) * weights) / np.mean(weights)

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

        #print(_rootU(upper), _rootL(lower), 'pivot')
        return lower + observed_stat, upper + observed_stat

    # Private methods

    def _weights(self, 
                 sample_stat,
                 nuisance,
                 score_cov,
                 log_densities):

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

        internal_sample = []
        for i in range(len(log_densities)):
            internal_sample.append(np.multiply.outer(sample_stat, score_cov[i]) + nuisance[i][None, :]) # these are now internal coordinates
        _lognum = self.opt_sampler.log_density(internal_sample, self.opt_sample)
        _logratio = _lognum - self._logden
        _logratio -= _logratio.max()

        return np.exp(_logratio)

def naive_confidence_intervals(target, observed, alpha=0.1):
    """
    Compute naive Gaussian based confidence
    intervals for target.
    Parameters
    ----------

    target : `targeted_sampler`
    observed : np.float
        A vector of observed data of shape `target.shape`
    alpha : float (optional)
        1 - confidence level.
    Returns
    -------
    intervals : np.float
        Gaussian based confidence intervals.
    """
    quantile = - ndist.ppf(alpha/float(2))
    LU = np.zeros((2, target.shape[0]))
    for j in range(target.shape[0]):
        sigma = np.sqrt(target.target_cov[j, j])
        LU[0,j] = observed[j] - sigma * quantile
        LU[1,j] = observed[j] + sigma * quantile
    return LU.T

def naive_pvalues(target, observed, parameter):
    pvalues = np.zeros(target.shape[0])
    for j in range(target.shape[0]):
        sigma = np.sqrt(target.target_cov[j, j])
        pval = ndist.cdf((observed[j]-parameter[j])/sigma)
        pvalues[j] = 2*min(pval, 1-pval)
    return pvalues
