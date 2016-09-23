from itertools import product
import numpy as np

from ..distributions.discrete_family import discrete_family
from ..sampling.langevin import projected_langevin

class multiple_views(object):

    '''
    Combine several views of a given data
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

        nviews = self.nviews = len(self.objectives)

        self.num_opt_var = 0
        self.opt_slice = []
        self.score_info = []

        for objective in self.objectives:
            score_ = objective.setup_sampler()
            self.score_info.append(score_)
            self.opt_slice.append(slice(self.num_opt_var, self.num_opt_var + objective.num_opt_var))
            self.num_opt_var += objective.num_opt_var

        self.observed_opt_state = np.zeros(self.num_opt_var)

        for i in range(nviews):
            self.observed_opt_state[self.opt_slice[i]] = self.objectives[i].observed_opt_state

        self.form_covariances = form_covariances

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

        return targeted_sampler(self,
                                target_info,
                                observed_target_state,
                                self.form_covariances,
                                target_set=target_set,
                                reference=reference)



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

        multi_view : `multiple_views`
           Instance of `multiple_views`. Attributes
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
        # these are arguments to target_bootstrap and score_bootstrap
        # nonparamteric bootstrap is np.random.choice(n, size=(n,), replace=True)
        # residual bootstrap might be X_E.dot(\bar{\beta}_E)
        # + np.random.choice(resid, size=(n,), replace=True)

        # if target_set is not None, we assume that
        # these coordinates (specified by a list of coordinates) of target
        # is assumed to be independent of the rest
        # the corresponding block of `target_cov` is zeroed out

        # we need these attributes of multi_view

        self.nviews = len(multi_view.objectives)
        self.opt_slice = multi_view.opt_slice
        self.objectives = multi_view.objectives

        self.observed_target_state = observed_target_state

        covariances = form_covariances(target_info, cross_terms=multi_view.score_info)

        self.target_cov = np.atleast_2d(covariances[0])
        # zero out some coordinates of target_cov
        # to enforce independence of target and null statistics

        if target_set is not None:
            null_set = set(range(self.target_cov.shape[0])).difference(target_set)
            for t, n in product(target_set, null_set):
                self.target_cov[t, n] = 0.
                self.target_cov[n, t] = 0.

        self.score_cov = covariances[1:]

        self.target_transform = []
        for i in range(self.nviews):
            self.target_transform.append(
                self.objectives[i].linear_decomposition(self.score_cov[i],
                                                        self.target_cov,
                                                        self.observed_target_state))

        self.target_inv_cov = np.linalg.inv(self.target_cov)
        # size of reference? should it only be target_set?
        if reference is None:
            reference = np.zeros(self.target_inv_cov.shape[0])
        self.reference_inv = self.target_inv_cov.dot(reference)

        # need to vectorize the state for Langevin

        self.overall_opt_slice = slice(0, multi_view.num_opt_var)
        self.target_slice = slice(multi_view.num_opt_var, 
                                  multi_view.num_opt_var + self.reference_inv.shape[0])
        self.keep_slice = self.target_slice

        # set the observed state

        self.observed_state = np.zeros(multi_view.num_opt_var + self.reference_inv.shape[0])
        self.observed_state[self.target_slice] = self.observed_target_state
        self.observed_state[self.overall_opt_slice] = multi_view.observed_opt_state

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

        target_state, opt_state = state[self.target_slice], state[self.overall_opt_slice]
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
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

        for i in range(self.nviews):
            target_grad_curr, opt_grad[self.opt_slice[i]] = \
                self.objectives[i].randomization_gradient(target_state, self.target_transform[i], opt_state[self.opt_slice[i]])
            target_grad += target_grad_curr.copy()

        target_grad = - target_grad
        target_grad += self.reference_inv - self.target_inv_cov.dot(target_state)
        full_grad[self.target_slice] = target_grad
        full_grad[self.overall_opt_slice] = -opt_grad

        return full_grad

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

        if stepsize is None:
            stepsize = 1. / self.crude_lipschitz()
        target_langevin = projected_langevin(self.observed_state.copy(),
                                             self.gradient,
                                             self.projection,
                                             stepsize)


        samples = []
        for i in range(ndraw + burnin):
            target_langevin.next()
            if i >= burnin:
                samples.append(target_langevin.state[self.keep_slice].copy())

        return samples

    def hypothesis_test(self,
                        test_stat,
                        observed_target,
                        ndraw=10000,
                        burnin=2000,
                        stepsize=None,
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

        observed_target : np.float
           Observed value of target estimate.
           Used in p-value calculation.

        ndraw : int
           How long a chain to return?

        burnin : int
           How many samples to discard?

        stepsize : float
           Stepsize for Langevin sampler. Defaults
           to a crude estimate based on the
           dimension of the problem.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------

        gradient : np.float

        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        samples = self.sample(ndraw, burnin, stepsize=stepsize)
        observed_stat = test_stat(observed_target)
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.cdf(0, observed_stat)

        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        else:
            return 2 * min(pval, 1 - pval)

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

# class conditional_targeted_sampler(targeted_sampler):
#     # condition on the optimization variables -- don't move them...

#     def __init__(self,
#                  multi_view,
#                  target_bootstrap,
#                  observed_target_state,
#                  target_set=None,
#                  reference=None):
#         targeted_sampler.__init__(self,
#                                   multi_view,
#                                   target_bootstrap,
#                                   observed_target_state,
#                                   target_set=target_set,
#                                   reference=reference)

#         # this is a hacky way to do things

#         self._full_state = self.observed_state.copy()
#         self._opt_state = self.observed_state[self.overall_opt_slice]
#         self.observed_state = self.observed_state[self.target_slice]
#         self.keep_slice = slice(None, None, None)

#     def gradient(self, state):
#         self._full_state[self.target_slice] = state
#         full_grad = targeted_sampler.gradient(self, self._full_state)
#         return full_grad[self.target_slice]

#     def projection(self, state):
#         return state

#     def crude_lipschitz(self):
#         result = np.linalg.svd(self.target_inv_cov)[1].max()
#         for transform, objective in zip(self.target_transform, self.objectives):
#             result += np.linalg.svd(transform[0])[1].max()**2 * objective.randomization.lipschitz
#         return result 
