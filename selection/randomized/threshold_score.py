import numpy as np
import regreg.api as rr

from .query import query
from .M_estimator import restricted_Mest

class threshold_score(query):

    def __init__(self, 
                 loss, 
                 threshold, 
                 randomization, 
                 active, 
                 inactive, 
                 beta_active=None,
                 solve_args={'min_its': 50, 'tol': 1.e-10}):
        """
        penalty is a group_lasso object that assigns weights to groups
        """

        query.__init__(self, randomization)

        # threshold could be a vector size inactive

        active_bool = np.zeros(loss.shape, np.bool)
        active_bool[active] = 1
        active = active_bool

        if np.array(threshold).shape in [(), (1,)]:
            threshold = np.ones(inactive.sum()) * threshold

        self.epsilon = 0.  # for randomized loss

        (self.loss,
         self.threshold,
         self.active,
         self.inactive,
         self.beta_active,
         self.randomization,
         self.solve_args) = (loss,
                             threshold,
                             active,
                             inactive,
                             beta_active,
                             randomization,
                             solve_args)

    def solve(self, nboot=2000):

        (loss,
         threshold,
         active,
         inactive,
         beta_active,
         randomization) = (self.loss,
                           self.threshold,
                           self.active,
                           self.inactive,
                           self.beta_active,
                           self.randomization)

        self._marginalize_subgradient = True # need to find a better place to set this...

        if beta_active is None:
            beta_active = self.beta_active = restricted_Mest(self.loss, active, solve_args=self.solve_args)

        self.randomize()

        beta_full = np.zeros(self.loss.shape)
        beta_full[active] = beta_active
        self._beta_full = beta_full

        inactive_score = self.loss.smooth_objective(beta_full, 'grad')[inactive]
        randomized_score = inactive_score + randomization.sample()

        # find the current active group, i.e.
        # subset of inactive that pass the threshold

        # TODO: make this test use group LASSO

        self.boundary = np.fabs(randomized_score) > threshold

        self.interior = ~self.boundary

        self.observed_score_state = inactive_score

        self.selection_variable = {'boundary_set': self.boundary}

        self._solved = True

        self.nboot = nboot
        self.ndim = self.loss.shape[0]

    def construct_weights(self, full_state):
        """
        marginalizing over the sub-gradient
        """

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        threshold = self.threshold
        weights = np.zeros_like(self.boundary, np.float)

        weights[self.boundary] = ((self.randomization._density(threshold[self.boundary] - full_state[self.boundary]) - self.randomization._density(-threshold[self.boundary] - full_state[self.boundary])) /
                                  (1 - self.randomization._cdf(threshold[self.boundary] - full_state[self.boundary]) + self.randomization._cdf(-threshold[self.boundary] - full_state[self.boundary])))


        weights[~self.boundary] = ((-self.randomization._density(threshold[~self.boundary] - full_state[~self.boundary]) + self.randomization._density(-threshold[~self.boundary] - full_state[~self.boundary])) /
                                   (self.randomization._cdf(threshold[~self.boundary] - full_state[~self.boundary]) - self.randomization._cdf(-threshold[~self.boundary] - full_state[~self.boundary])))

        return weights ## tested

    def setup_sampler(self):

        # must set observed_opt_state, opt_transform and score_transform

        p = self.boundary.shape[0]  # shorthand
        self.num_opt_var = 0
        self.opt_transform = (None, None)
        self.observed_opt_state = np.array([])
        _score_linear_term = -np.identity(p)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self._setup = True

    def projection(self, opt_state):
        """
        Full projection for Langevin.
        The state here will be only the state of the optimization variables.
        for now, groups are singletons
        """
        return opt_state

