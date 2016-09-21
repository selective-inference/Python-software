from itertools import product
import numpy as np

from ..distributions.discrete_family import discrete_family
from ..sampling.langevin import projected_langevin

class multiple_views(object):

    def __init__(self, objectives):

        self.objectives = objectives

    def solve(self):
        for objective in self.objectives:
            # maybe just check if they have been solved
            # randomize first?
            objective.solve()

    def setup_sampler(self, form_covariances):

        nviews = self.nviews = len(self.objectives)

        self.num_opt_var = 0
        self.opt_slice = []
        self.score_info = []

        for objective in self.objectives:
            score_ = objective.setup_sampler() # shouldn't have to refit all the time as this function does
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
                     boot_size,
                     target_alpha,
                     target_set=None,
                     reference=None,
                     constructor=None):

        if constructor is None:
            constructor = targeted_sampler

        return constructor(self,
                           target_info,
                           observed_target_state,
                           self.form_covariances,
                           boot_size,
                           target_alpha,
                           target_set=target_set,
                           reference=reference)



class targeted_sampler(object):

    # make one of these for each hypothesis test

    def __init__(self,
                 multi_view,
                 target_info,
                 observed_target_state,
                 form_covariances,
                 boot_size,
                 target_alpha,
                 target_set=None,
                 reference=None):

        # sampler will draw samples for bootstrap
        # these are arguments to target_bootstrap and score_bootstrap
        # nonparamteric bootstrap is np.random.choice(n, size=(n,), replace=True)
        # residual bootstrap might be X_E.dot(\bar{\beta}_E) + np.random.choice(resid, size=(n,), replace=True)

        # if target_set is not None, we assume that these coordinates (specified by a list of coordinates) of target
        # is assumed to be independent of the rest
        # the corresponding block of `target_cov` is zeroed out

        # we need these attributes of multi_view

        self.nviews = len(multi_view.objectives)
        self.opt_slice = multi_view.opt_slice
        self.objectives = multi_view.objectives

        #for bootstrap
        self.boot_size = boot_size
        self.target_alpha = target_alpha
        self.inv_mat = np.linalg.inv(np.dot(self.target_alpha, self.target_alpha.T))


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

        self.target_inv_cov = np.linalg.inv(self.target_cov)

        # size of reference? should it only be target_set?
        if reference is None:
            reference = np.zeros(self.target_inv_cov.shape[0])

        self.target_transform = []
        self.boot_transform = [] # bootstrap

        for i in range(self.nviews):
            self.target_transform.append(self.objectives[i].linear_decomposition(self.score_cov[i], 
                                                                                 self.target_cov,
                                                                                 self.observed_target_state))

            self.boot_transform.append(self.objectives[i].boot_decomposition(target_alpha,
                                                                             reference,
                                                                             self.score_cov[i],
                                                                             self.target_cov,
                                                                             self.observed_target_state))

        self.reference_inv = self.target_inv_cov.dot(reference)

        # need to vectorize the state for Langevin

        self.overall_opt_slice = slice(0, multi_view.num_opt_var)
        self.target_slice = slice(multi_view.num_opt_var, multi_view.num_opt_var + self.reference_inv.shape[0])
        self.keep_slice = self.target_slice

        # set the observed state

        self.observed_state = np.zeros(multi_view.num_opt_var + self.reference_inv.shape[0])
        self.observed_state[self.target_slice] = self.observed_target_state
        self.observed_state[self.overall_opt_slice] = multi_view.observed_opt_state

        # set the observed state for bootstrap

        self.boot_slice = slice(multi_view.num_opt_var, multi_view.num_opt_var + self.boot_size)
        self.boot_observed_state = np.zeros(multi_view.num_opt_var + self.boot_size)
        self.boot_observed_state[self.boot_slice] = np.ones(self.boot_size)
        self.boot_observed_state[self.overall_opt_slice] = multi_view.observed_opt_state



    def projection(self, state):
        target_state, opt_state = state[self.target_slice], state[self.overall_opt_slice] 
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
            new_opt_state[self.opt_slice[i]] = self.objectives[i].projection(opt_state[self.opt_slice[i]])
        state[self.overall_opt_slice] = new_opt_state
        return state


    def gradient(self, state):

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


    def boot_projection(self, state):
        boot_state, opt_state = state[self.boot_slice], state[self.overall_opt_slice]
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
            new_opt_state[self.opt_slice[i]] = self.objectives[i].projection(opt_state[self.opt_slice[i]])
        state[self.overall_opt_slice] = new_opt_state
        return state

    def boot_gradient(self, state):

        boot_state, opt_state = state[self.boot_slice], state[self.overall_opt_slice]
        boot_grad, opt_grad = np.zeros_like(boot_state), np.zeros_like(opt_state)
        full_grad = np.zeros_like(state)

        # randomization_gradient are gradients of a CONVEX function

        for i in range(self.nviews):
            boot_grad_curr, opt_grad[self.opt_slice[i]] = \
                self.objectives[i].randomization_gradient(boot_state, self.boot_transform[i],
                                                          opt_state[self.opt_slice[i]])
            boot_grad += boot_grad_curr.copy()

        boot_grad = -boot_grad
        boot_grad -= boot_state
        #boot_grad -= np.dot(np.dot(self.target_alpha.T, self.inv_mat), self.target_alpha.dot(boot_state))
        #boot_grad -= np.dot(np.dot(self.target_alpha.T, self.target_inv_cov), self.target_alpha.dot(boot_state))

        full_grad[self.boot_slice] = boot_grad
        full_grad[self.overall_opt_slice] = -opt_grad

        return full_grad


    def sample(self, ndraw, burnin, stepsize = None):
        """
        assumes setup_sampler has been called
        """
        if stepsize is None:
            #stepsize = 1. /self.boot_size
            stepsize = 1. / self.observed_state.shape[0]
        target_langevin = projected_langevin(self.observed_state.copy(),
                                             self.gradient,
                                             self.projection,
                                             stepsize)

        bootstrap_langevin = projected_langevin(self.boot_observed_state.copy(),
                                                self.boot_gradient,
                                                self.boot_projection,
                                                stepsize)

        samples = []
        for i in range(ndraw + burnin):
            bootstrap_langevin.next()
            if (i >= burnin):
                samples.append(bootstrap_langevin.state[self.boot_slice].copy())

        return samples

    def hypothesis_test(self, 
                        test_stat, 
                        observed_target, 
                        ndraw=10000,
                        burnin=2000,
                        stepsize=None,
                        alternative='twosided'):

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


    def boot_hypothesis_test(self,
                        test_stat,
                        observed_value,
                        ndraw = 10000,
                        burnin = 2000,
                        stepsize=None,
                        alternative='twosided'):


        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        samples = self.sample(ndraw, burnin, stepsize=stepsize)
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.cdf(0, observed_value)

        print pval
        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        else:
            return 2 * min(pval, 1 - pval)


    def crude_lipschitz(self):
        result = np.linalg.svd(self.target_inv_cov)[1].max()
        for transform, objective in zip(self.target_transform, self.objectives):
            result += np.linalg.svd(transform[0])[1].max()**2 * objective.randomization.lipschitz
            result += np.linalg.svd(objective.score_transform[0])[1].max()**2 * objective.randomization.lipschitz
        return result 

class conditional_targeted_sampler(targeted_sampler):
    # condition on the optimization variables -- don't move them...

    def __init__(self,
                 multi_view,
                 target_bootstrap,
                 observed_target_state,
                 target_set=None,
                 reference=None):
        targeted_sampler.__init__(self,
                                  multi_view,
                                  target_bootstrap,
                                  observed_target_state,
                                  target_set=target_set,
                                  reference=reference)

        # this is a hacky way to do things

        self._full_state = self.observed_state.copy()
        self._opt_state = self.observed_state[self.overall_opt_slice]
        self.observed_state = self.observed_state[self.target_slice]
        self.keep_slice = slice(None, None, None)

    def gradient(self, state):
        self._full_state[self.target_slice] = state
        full_grad = targeted_sampler.gradient(self, self._full_state)
        return full_grad[self.target_slice]

    def projection(self, state):
        return state

    def crude_lipschitz(self):
        result = np.linalg.svd(self.target_inv_cov)[1].max()
        for transform, objective in zip(self.target_transform, self.objectives):
            result += np.linalg.svd(transform[0])[1].max()**2 * objective.randomization.lipschitz
        return result 
