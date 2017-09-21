from itertools import product
import numpy as np

from regreg.affine import power_L

from ..distributions.api import discrete_family, intervals_from_sample
from ..sampling.langevin import projected_langevin
from .reconstruction import reconstruct_full_from_data, reconstruct_internal

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
                 target_set=None,
                 parametric=False):

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

        parametric : bool
           Use parametric covariance estimate?

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

        self.total_randomization_length = multi_view.total_randomization_length
        self.randomization_slice = multi_view.randomization_slice

        self.score_cov = []
        target_cov_sum = 0
        for i in range(self.nqueries):
            if parametric == False:
                target_cov, cross_cov = multi_view.form_covariances(target_info,  
                                                                    cross_terms=[multi_view.score_info[i]],
                                                                    nsample=multi_view.nboot[i])
            else:
                target_cov, cross_cov = multi_view.form_covariances(target_info, 
                                                                    cross_terms=[multi_view.score_info[i]])

            target_cov_sum += target_cov
            self.score_cov.append(cross_cov)

        self.target_cov = target_cov_sum / self.nqueries

        # XXX we're not really using this target_set in our tests

        # zero out some coordinates of target_cov
        # to enforce independence of target and null statistics

        if target_set is not None:
            null_set = set(range(self.target_cov.shape[0])).difference(target_set)
            for t, n in product(target_set, null_set):
                self.target_cov[t, n] = 0.
                self.target_cov[n, t] = 0.

        self.target_transform = []

        for i in range(self.nqueries):
            self.target_transform.append(
                self.objectives[i].linear_decomposition(self.score_cov[i],
                                                        self.target_cov,
                                                        self.observed_target_state))

        self.target_cov = np.atleast_2d(self.target_cov)
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

        # added for the reconstruction map in case we marginalize over optimization variables

        randomization_length_total = 0
        self.randomization_slice = []
        for i in range(self.nqueries):
            self.randomization_slice.append(
                slice(randomization_length_total, randomization_length_total + self.objectives[i].ndim))
            randomization_length_total += self.objectives[i].ndim

        self.randomization_length_total = randomization_length_total

    def set_reference(self, reference):
        self._reference = np.atleast_1d(reference)
        self._reference_inv = self.target_inv_cov.dot(self.reference).flatten()

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

            randomization_state = reconstruct_full_from_data(self.objectives[i],
                                                             target_state, 
                                                             self.target_transform[i], 
                                                             opt_state[self.opt_slice[i]])

            internal_state = reconstruct_internal(target_state, self.target_transform[i])
            grad = self.objectives[i].grad_log_density(internal_state, opt_state[self.opt_slice[i]]) 
            target_linear, target_offset = self.target_transform[i]
            opt_linear, opt_offset = self.objectives[i].opt_transform
            if target_linear is not None:
                target_grad += target_linear.T.dot(grad)
            if opt_linear is not None:
                opt_grad[self.opt_slice[i]] = opt_offset.T.dot(grad)

        target_grad = -target_grad
        target_grad += self._reference_inv - self.target_inv_cov.dot(target_state)
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
        lipschitz = power_L(self.target_inv_cov)
        for transform, objective in zip(self.target_transform, self.objectives):
            lipschitz += power_L(transform[0])**2 * objective.randomization.lipschitz
            lipschitz += power_L(objective.score_transform[0])**2 * objective.randomization.lipschitz
        return lipschitz


    def reconstruct(self, state):
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
        reconstructed = np.zeros((state.shape[0], self.total_randomization_length))

        for i in range(self.nqueries):
            reconstructed[:, self.randomization_slice[i]] = reconstruct_full_from_data(self.objectives[i],
                                                                                       target_state,
                                                                                       self.target_transform[i],
                                                                                       opt_state[:, self.opt_slice[i]])

        return np.squeeze(reconstructed)

    def log_density(self, state):
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

        reconstructed = self.reconstruct(state)
        value = np.zeros(reconstructed.shape[0])

        for i in range(self.nqueries):
            log_dens = self.objectives[i].randomization.log_density
            value += log_dens(reconstructed[:,self.opt_slice[i]])
        return np.squeeze(value)

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

            randomization_state = reconstruct_full_from_data(self.objectives[i],
                                                             boot_state, 
                                                             self.boot_transform[i], 
                                                             opt_state[self.opt_slice[i]])

            internal_state = reconstruct_internal(boot_state, self.boot_transform[i])
            grad = self.objectives[i].grad_log_density(internal_state, opt_state[self.opt_slice[i]])
            boot_linear, boot_offset = self.boot_transform[i]
            opt_linear, opt_offset = self.objectives[i].opt_transform
            if boot_linear is not None:
                boot_grad += boot_linear.T.dot(grad)
            if opt_linear is not None:
                opt_grad[self.opt_slice[i]] = opt_offset.T.dot(grad)

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
