from itertools import product
import numpy as np
from scipy.stats import norm as ndist
from scipy.optimize import bisect

from ..distributions.api import discrete_family, intervals_from_sample
from ..sampling.langevin import projected_langevin

class query(object):

    def __init__(self, randomization):

        self.randomization = randomization
        self._solved = False
        self._randomized = False
        self._setup = False

    # Methods reused by subclasses

    def randomize(self):

        if not self._randomized:
            self.randomized_loss = self.randomization.randomize(self.loss, self.epsilon)
        self._randomized = True

    def randomization_gradient(self, data_state, data_transform, opt_state):
        """
        Randomization derivative at full state.
        """

        # reconstruction of randoimzation omega

        opt_linear, opt_offset = self.opt_transform
        data_linear, data_offset = data_transform
        data_piece = data_linear.dot(data_state) + data_offset

        # value of the randomization omega

        if opt_linear is not None: # this can happen if we marginalize all of omega!
            opt_piece = opt_linear.dot(opt_state) + opt_offset
            full_state = (data_piece + opt_piece)
        else:
            full_state = data_piece

        # gradient of negative log density of randomization at omega

        if self._marginalize_subgradient==False:
            randomization_derivative = self.randomization.gradient(full_state)
        else:
            randomization_derivative = self.construct_weights(full_state)

        # chain rule for data, optimization parts

        data_grad = data_linear.T.dot(randomization_derivative)
        if opt_linear is not None:
            opt_grad = opt_linear.T.dot(randomization_derivative)
        else:
            opt_grad = None
        return data_grad, opt_grad #- self.grad_log_jacobian(opt_state)

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

    def reconstruction_map(self, data_state, data_transform, opt_state):

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        # reconstruction of randoimzation omega

        data_state = np.atleast_2d(data_state)
        opt_state = np.atleast_2d(opt_state)

        opt_linear, opt_offset = self.opt_transform
        data_linear, data_offset = data_transform
        data_piece = data_linear.dot(data_state.T) + data_offset[:, None]
        opt_piece = opt_linear.dot(opt_state.T) + opt_offset[:, None]

        # value of the randomization omega

        return (data_piece + opt_piece).T

    def log_density(self, data_state, data_transform, opt_state):

        full_data = self.reconstruction_map(data_state, data_transform, opt_state)
        return self.randomization.log_density(full_data)

    # Abstract methods to be
    # implemented by subclasses

    def grad_log_jacobian(self, opt_state):
        """
        log_jacobian depends only on data through
        Hessian at \bar{\beta}_E which we
        assume is close to Hessian at \bar{\beta}_E^*
        """
        # needs to be implemented for group lasso
        return 0.

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

        for objective in self.objectives:
            score_ = objective.setup_sampler()
            self.score_info.append(score_)

    def setup_opt_state(self):
        self.num_opt_var = 0
        self.opt_slice = []

        for objective in self.objectives:
            self.opt_slice.append(slice(self.num_opt_var, self.num_opt_var + objective.num_opt_var))
            self.num_opt_var += objective.num_opt_var

        self.observed_opt_state = np.zeros(self.num_opt_var)
        for i in range(len(self.objectives)):
            self.observed_opt_state[self.opt_slice[i]] = self.objectives[i].observed_opt_state

    def setup_target(self,
                     target_info,
                     observed_target_state,
                     reference=None,
                     target_set=None):

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
                                reference=reference)

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

class targeted_sampler(object):

    '''
    Object to sample from target of a selective sampler.
    '''

    def __init__(self,
                 multi_view,
                 target_info,
                 observed_target_state,
                 form_covariances,
                 reference=None,
                 target_set=None):

        '''
        Parameters
        ----------
        multi_view : `multiple_queries`
           Instance of `multiple_queries`. Attributes
           `objectives`, `score_info` are key
           attributed. (Should maybe change constructor
           to reflect only what is needed.)
        target_info : object
           Passed as first argument to `self.form_covariances`.
        observed_target_state : np.float
           Observed value of the target estimator.
        form_covariances : callable
           Used in linear decomposition of each score
           and the target.
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
        The callable `form_covariances`
        should accept `target_info` as first argument
        and a keyword argument `cross_terms` which
        correspond to the `score_info` of each
        objective of `multi_view`. This used in
        a linear decomposition of each score into
        a piece correlated with `target` and
        an independent piece.
        The independent piece is treated as a
        nuisance parameter and conditioned on
        (i.e. is fixed within the sampler).
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

        # we need these attributes of multi_view

        self.nqueries = len(multi_view.objectives)
        self.opt_slice = multi_view.opt_slice
        self.objectives = multi_view.objectives

        self.observed_target_state = observed_target_state
        self.shape = observed_target_state.shape

        covariances = multi_view.form_covariances(target_info, cross_terms=multi_view.score_info)
        self.target_cov = np.atleast_2d(covariances[0])

        # XXX we're not really using this target_set in our tests

        # zero out some coordinates of target_cov
        # to enforce independence of target and null statistics

        if target_set is not None:
            null_set = set(range(self.target_cov.shape[0])).difference(target_set)
            for t, n in product(target_set, null_set):
                self.target_cov[t, n] = 0.
                self.target_cov[n, t] = 0.

        self.score_cov = covariances[1:]

        self.target_transform = []
        for i in range(self.nqueries):
            self.target_transform.append(
                self.objectives[i].linear_decomposition(self.score_cov[i],
                                                        self.target_cov,
                                                        self.observed_target_state))

        self.target_inv_cov = np.linalg.inv(self.target_cov)
        # size of reference? should it only be target_set?
        if reference is None:
            reference = np.zeros(self.target_inv_cov.shape[0])
        self.reference = reference

        # need to vectorize the state for Langevin

        self.overall_opt_slice = slice(0, multi_view.num_opt_var)
        self.target_slice = slice(multi_view.num_opt_var,
                                  multi_view.num_opt_var + self._reference_inv.shape[0])
        self.keep_slice = self.target_slice

        # set the observed state

        self.observed_state = np.zeros(multi_view.num_opt_var + self._reference_inv.shape[0])
        self.observed_state[self.target_slice] = self.observed_target_state
        self.observed_state[self.overall_opt_slice] = multi_view.observed_opt_state

    def set_reference(self, reference):
        self._reference = np.atleast_1d(reference)
        self._reference_inv = self.target_inv_cov.dot(self.reference)

    def get_reference(self):
        return self._reference

    reference = property(get_reference, set_reference)

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

        opt_state = state[self.overall_opt_slice]
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nqueries):
            new_opt_state[self.opt_slice[i]] = self.objectives[i].projection(opt_state[self.opt_slice[i]])
        state[self.overall_opt_slice] = new_opt_state
        return state

    def gradient(self, state):
        '''
        Gradient of log-density at current state.
        Parameters
        ----------
        state : np.float
           State of sampler made up of `(target, opt_vars)`.
        Returns
        -------
        gradient : np.float
        '''

        target_state, opt_state = state[self.target_slice], state[self.overall_opt_slice]
        target_grad, opt_grad = np.zeros_like(target_state), np.zeros_like(opt_state)
        full_grad = np.zeros_like(state)

        # randomization_gradient are gradients of a CONVEX function

        for i in range(self.nqueries):
            target_grad_curr, opt_grad[self.opt_slice[i]] = \
                self.objectives[i].randomization_gradient(target_state, self.target_transform[i], opt_state[self.opt_slice[i]])
            target_grad += target_grad_curr.copy()

        target_grad = - target_grad
        target_grad += self._reference_inv.flatten() - self.target_inv_cov.dot(target_state)
        full_grad[self.target_slice] = target_grad
        full_grad[self.overall_opt_slice] = -opt_grad

        return full_grad

    def sample(self, ndraw, burnin, stepsize=None, keep_opt=False):
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
            stepsize = 1. / self.crude_lipschitz()

        if keep_opt:
            keep_slice = slice(None, None, None)
        else:
            keep_slice = self.keep_slice

        target_langevin = projected_langevin(self.observed_state.copy(),
                                             self.gradient,
                                             self.projection,
                                             stepsize)

        samples = []
        for i in range(ndraw + burnin):
            target_langevin.next()
            if (i >= burnin):
                samples.append(target_langevin.state[keep_slice].copy())

        return np.asarray(samples)

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

        sample_test_stat = np.squeeze(np.array([test_stat(x) for x in sample]))

        if parameter is None:
            parameter = self.reference

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
                             observed,
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

        nactive = observed.shape[0]
        intervals_instance = intervals_from_sample(self.reference,
                                                   sample,
                                                   observed,
                                                   self.target_cov)

        return intervals_instance.confidence_intervals_all(level=level)

    def coefficient_pvalues(self,
                            observed,
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
            parameter = np.zeros(self.shape)

        nactive = observed.shape[0]
        intervals_instance = intervals_from_sample(self.reference,
                                                   sample,
                                                   observed,
                                                   self.target_cov)

        pval = intervals_instance.pivots_all(parameter)

        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        else:
            return 2 * np.minimum(pval, 1 - pval)

    def crude_lipschitz(self):
        """
        A crude Lipschitz constant for the
        gradient of the log-density.
        Returns
        -------
        lipschitz : float

        """
        lipschitz = np.linalg.svd(self.target_inv_cov)[1].max()
        for transform, objective in zip(self.target_transform, self.objectives):
            lipschitz += np.linalg.svd(transform[0])[1].max()**2 * objective.randomization.lipschitz
            lipschitz += np.linalg.svd(objective.score_transform[0])[1].max()**2 * objective.randomization.lipschitz
        return lipschitz


    def reconstruction_map(self, state):
        '''
        Reconstruction of randomization at current state.
        Parameters
        ----------
        state : np.float
           State of sampler made up of `(target, opt_vars)`.
           Can be array with each row a state.
        Returns
        -------
        reconstructed : np.float
           Has shape of `opt_vars` with same number of rows
           as `state`.

        '''

        state = np.atleast_2d(state)
        if len(state.shape) > 2:
            raise ValueError('expecting at most 2-dimensional array')

        target_state, opt_state = state[:,self.target_slice], state[:,self.overall_opt_slice]
        reconstructed = np.zeros_like(opt_state)

        for i in range(self.nqueries):
            reconstructed[:, self.opt_slice[i]] = self.objectives[i].reconstruction_map(target_state,
                                                                                        self.target_transform[i],
                                                                                        opt_state[:,self.opt_slice[i]])
        return np.squeeze(reconstructed)

    def log_randomization_density(self, state):
        '''
        Log of randomization density at current state.
        Parameters
        ----------
        state : np.float
           State of sampler made up of `(target, opt_vars)`.
           Can be two-dimensional with each row a state.
        Returns
        -------
        density : np.float
            Has number of rows as `state` if 2-dimensional.
        '''

        reconstructed = self.reconstruction_map(state)
        value = np.zeros(reconstructed.shape[0])

        for i in range(self.nqueries):
            log_dens = self.objectives[i].randomization.log_density
            value += log_dens(reconstructed[:,self.opt_slice[i]])
        return np.squeeze(value)

    def hypothesis_test_translate(self,
                                  sample,
                                  test_stat,
                                  observed_target,
                                  parameter=None,
                                  alternative='twosided'):

        '''
        Carry out a hypothesis test
        based on the distribution of the
        residual `observed_target - target`
        sampled at `self.reference`.
        Parameters
        ----------
        sample : np.array
           Sample of target and optimization variables drawn at `self.reference`.
        test_stat : callable
           Test statistic to evaluate on sample from
           selective distribution.
        observed_target : np.float
           Observed value of target estimate.
           Used in p-value calculation.
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

        _intervals = translate_intervals(self,
                                         sample,
                                         observed_target)

        if parameter is None:
            parameter = self.reference

        return _intervals.pivot(test_stat,
                                parameter,
                                alternative=alternative)


    def confidence_intervals_translate(self,
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
            sample = self.sample(ndraw, burnin, stepsize=stepsize, keep_opt=True)

        _intervals = translate_intervals(self,
                                         sample,
                                         observed_target)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            limits.append(_intervals.confidence_interval(keep, level=level))

        return np.array(limits)

    def coefficient_pvalues_translate(self,
                                      observed_target,
                                      parameter=None,
                                      ndraw=10000,
                                      burnin=2000,
                                      stepsize=None,
                                      sample=None,
                                      alternative='twosided'):
        '''
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
            P values for each coefficient.

        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample = self.sample(ndraw, burnin, stepsize=stepsize, keep_opt=True)

        if parameter is None:
            parameter = np.zeros_like(observed_target)

        _intervals = translate_intervals(self,
                                         sample,
                                         observed_target)

        pvalues = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.

            _parameter = self.reference.copy()
            _parameter[i] = parameter[i]
            pvalues.append(_intervals.pivot(lambda x: keep.dot(x),
                                            _parameter,
                                            alternative=alternative))

        return np.array(pvalues)

class bootstrapped_target_sampler(targeted_sampler):

    # make one of these for each hypothesis test

    def __init__(self,
                 multi_view,
                 target_info,
                 observed_target_state,
                 target_alpha,
                 target_set=None,
                 reference=None,
                 boot_size=None):

        # sampler will draw bootstrapped weights for the target

        if boot_size is None:
            boot_size = target_alpha.shape[1]

        targeted_sampler.__init__(self, multi_view,
                                  target_info,
                                  observed_target_state,
                                  target_set,
                                  reference)
        # for bootstrap

        self.boot_size = boot_size
        self.target_alpha = target_alpha
        self.boot_transform = []


        for i in range(self.nqueries):
            composition_linear_part, composition_offset = self.objectives[i].linear_decomposition(self.score_cov[i],
                                                                                                  self.target_cov,
                                                                                                  self.observed_target_state)
            boot_linear_part = np.dot(composition_linear_part, target_alpha)
            boot_offset = composition_offset + np.dot(composition_linear_part, self.reference).flatten()
            self.boot_transform.append((boot_linear_part, boot_offset))

        # set the observed state for bootstrap

        self.boot_slice = slice(multi_view.num_opt_var, multi_view.num_opt_var + self.boot_size)
        self.observed_state = np.zeros(multi_view.num_opt_var + self.boot_size)
        self.observed_state[self.boot_slice] = np.ones(self.boot_size)
        self.observed_state[self.overall_opt_slice] = multi_view.observed_opt_state


    def gradient(self, state):

        boot_state, opt_state = state[self.boot_slice], state[self.overall_opt_slice]
        boot_grad, opt_grad = np.zeros_like(boot_state), np.zeros_like(opt_state)
        full_grad = np.zeros_like(state)

        # randomization_gradient are gradients of a CONVEX function

        for i in range(self.nqueries):
            boot_grad_curr, opt_grad[self.opt_slice[i]] = \
                self.objectives[i].randomization_gradient(boot_state, self.boot_transform[i],
                                                          opt_state[self.opt_slice[i]])
            boot_grad += boot_grad_curr.copy()

        boot_grad = -boot_grad
        boot_grad -= boot_state

        full_grad[self.boot_slice] = boot_grad
        full_grad[self.overall_opt_slice] = -opt_grad

        return full_grad

    def sample(self, ndraw, burnin, stepsize = None, keep_opt=False):
        if stepsize is None:
            stepsize = 1. / self.observed_state.shape[0]

        bootstrap_langevin = projected_langevin(self.observed_state.copy(),
                                                self.gradient,
                                                self.projection,
                                                stepsize)
        if keep_opt:
            boot_slice = slice(None, None, None)
        else:
            boot_slice = self.boot_slice

        samples = []
        for i in range(ndraw + burnin):
            bootstrap_langevin.next()
            if (i >= burnin):
                samples.append(bootstrap_langevin.state[boot_slice].copy())
        samples = np.asarray(samples)

        if keep_opt:
            target_samples = samples[:,self.boot_slice].dot(self.target_alpha.T) + self.reference[None, :]
            opt_sample0 = samples[0,self.overall_opt_slice]
            result = np.zeros((samples.shape[0], opt_sample0.shape[0] + target_samples.shape[1]))
            result[:,self.overall_opt_slice] = samples[:,self.overall_opt_slice]
            result[:,self.target_slice] = target_samples
            return result
        else:
            target_samples = samples.dot(self.target_alpha.T) + self.reference[None, :]
            return target_samples

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


class translate_intervals(object): # intervals_from_sample):

    """
    Location family based intervals... (cryptic)
    randomization density should be `g` composed with the affine
    mapping and take an argument like one row of sample
    target_linear is the linear part of the affine mapping with
    respect to target
    weights for a given candidate will look like
          randomization_density(sample + (candidate, 0, 0) - (reference, 0, 0)) /
          randomization_density(sample)
    if the samples are samples of \bar{\beta}. if we have samples of
    \Delta from our reference, then the weights will look like
    randomization_density(sample + (candidate, 0, 0))
    randomization_density(sample + (reference, 0, 0))
    WE ARE ASSUMING sample is sampled from targeted_sampler.reference
    """

    def __init__(self,
                 targeted_sampler,
                 sample,
                 observed):
        self.targeted_sampler = targeted_sampler
        self.observed = observed.copy() # this is our observed unpenalized estimator
        self._logden = targeted_sampler.log_randomization_density(sample)
        self._delta = sample.copy()
        self._delta[:, targeted_sampler.target_slice] -= targeted_sampler.reference[None, :]

    def pivot(self,
              test_statistic,
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

        observed_delta = self.observed - candidate
        observed_stat = test_statistic(observed_delta)

        candidate_sample, weights = self._weights(candidate)
        #sample_stat = np.array([test_statistic(s) for s in candidate_sample[:, self.targeted_sampler.target_slice]])
        sample_stat = np.array([test_statistic(s) for s in self._delta[:, self.targeted_sampler.target_slice]])

        pivot = np.mean((sample_stat <= observed_stat) * weights) / np.mean(weights)

        if alternative == 'twosided':
            return 2 * min(pivot, 1 - pivot)
        elif alternative == 'less':
            return pivot
        else:
            return 1 - pivot

    def confidence_interval(self, linear_func, level=0.95, how_many_sd=20):

        target_delta = self._delta[:,self.targeted_sampler.target_slice]
        projected_delta = target_delta.dot(linear_func)
        projected_observed = self.observed.dot(linear_func)

        delta_min, delta_max = projected_delta.min(), projected_delta.max()

        _norm = np.linalg.norm(linear_func)
        grid_min, grid_max = -how_many_sd * np.std(projected_delta), how_many_sd * np.std(projected_delta)

        reference = self.targeted_sampler.reference

        def _rootU(gamma):
            return self.pivot(lambda x: linear_func.dot(x),
                              reference + gamma * linear_func / _norm**2,
                              alternative='less') - (1 - level) / 2.


        def _rootL(gamma):
            return self.pivot(lambda x: linear_func.dot(x),
                              reference + gamma * linear_func / _norm**2,
                              alternative='less') - (1 + level) / 2.

        upper = bisect(_rootU, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))
        lower = bisect(_rootL, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))

        return lower + projected_observed, upper + projected_observed

    # Private methods

    def _weights(self, candidate):

        candidate_sample = self._delta.copy()
        candidate_sample[:, self.targeted_sampler.target_slice] += candidate[None, :]
        _lognum = self.targeted_sampler.log_randomization_density(candidate_sample)

        _logratio = _lognum - self._logden
        _logratio -= _logratio.max()

        return candidate_sample, np.exp(_logratio)


