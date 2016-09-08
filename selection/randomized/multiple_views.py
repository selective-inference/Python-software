from itertools import product
import numpy as np
from .glm_boot import bootstrap_cov

class multiple_views(object):

    def __init__(self, objectives):

        self.objectives = objectives
        self.nviews = len(self.objectives)

    def solve(self):
        for i in range(self.nviews):
            # maybe just check if they have been solved
            # randomize first?
            self.objectives[i].solve()

    def setup_sampler(self, 
                      sampler, 
                      target_bootstrap, 
                      observed_target_state, 
                      target_set=None,
                      reference=None):

        # sampler will draw samples for bootstrap
        # these are arguments to target_bootstrap and score_bootstrap
        # nonparamteric bootstrap is np.random.choice(n, size=(n,), replace=True)
        # residual bootstrap might be X_E.dot(\bar{\beta}_E) + np.random.choice(resid, size=(n,), replace=True)

        # if target_set is not None, we assume that these coordinates (specified by a list of coordinates) of target
        # is assumed to be independent of the rest
        # the corresponding block of `target_cov` is zeroed out

        self.num_opt_var = 0
        self.opt_slice = []
        self.score_bootstrap = []

        for i in range(self.nviews):
            score_bootstrap = self.objectives[i].setup_sampler()
            self.score_bootstrap.append(score_bootstrap)
            self.opt_slice.append(slice(self.num_opt_var, self.num_opt_var+self.objectives[i].num_opt_var))
            self.num_opt_var += self.objectives[i].num_opt_var

        self.observed_opt_state = np.zeros(self.num_opt_var)

        for i in range(self.nviews):
            self.observed_opt_state[self.opt_slice[i]] = self.objectives[i].observed_opt_state

        # now setup conditioning

        self.observed_target_state = observed_target_state

        covariances = bootstrap_cov(sampler, target_bootstrap, cross_terms=self.score_bootstrap)
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
            self.target_transform.append(self.objectives[i].condition(self.score_cov[i], 
                                                                      self.target_cov,
                                                                      self.observed_target_state))
        self.target_inv_cov = np.linalg.inv(self.target_cov)
        if reference is None:
            reference = np.zeros(self.target_inv_cov.shape[0])
        self.reference_inv = self.target_inv_cov.dot(reference)

        # need to vectorize the state for Langevin

        self.overall_opt_slice = slice(0, self.num_opt_var)
        self.target_slice = slice(self.num_opt_var, self.num_opt_var + self.reference_inv.shape[0])

        # set the observed state

        self.observed_state = np.zeros(self.num_opt_var + self.reference_inv.shape[0])
        self.observed_state[self.target_slice] = self.observed_target_state
        self.observed_state[self.overall_opt_slice] = self.observed_opt_state

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


