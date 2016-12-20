import numpy as np
import regreg.api as rr

from .query import query
from .M_estimator import restricted_Mest


class threshold_score(query):
    def __init__(self, loss, threshold, randomization, active, inactive, beta_active=None,
                 solve_args={'min_its': 50, 'tol': 1.e-10}):
        """
        penalty is a group_lasso object that assigns weights to groups
        """

        query.__init__(self, randomization)

        # threshold could be a vector size inactive

        active_bool = np.zeros(loss.shape, np.bool)
        active_bool[active] = 1
        active = active_bool
        inactive = ~active

        if type(threshold) == type(0.):
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

    def solve(self):

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

        if beta_active is None:
            beta_active = self.beta_active = restricted_Mest(self.loss, active, solve_args=self.solve_args)

        self.randomize()

        beta_full = np.zeros(self.loss.shape)
        beta_full[active] = beta_active

        inactive_score = self.loss.smooth_objective(beta_full, 'grad')[inactive]
        randomized_score = self.loss.smooth_objective(beta_full, 'grad')[inactive]

        # find the current active group, i.e.
        # subset of inactive that pass the threshold

        # TODO: make this test use group LASSO

        self.boundary = np.fabs(randomized_score) > threshold
        self.boundary_signs = np.sign(randomized_score)[self.boundary]
        self.interior = ~self.boundary

        self.observed_overshoot = self.boundary_signs * (inactive_score[self.boundary] - threshold[self.boundary])
        self.observed_below_thresh = inactive_score[self.interior]
        self.observed_score_state = inactive_score

        self.selection_variable = {'boundary_set': self.boundary,
                                   'boundary_signs': self.boundary_signs}

        self._solved = True

        self.num_opt_var = self.boundary.shape[0]

    def setup_sampler(self):

        # must set observed_opt_state, opt_transform and score_transform

        p = self.boundary.shape[0]  # shorthand
        self.observed_opt_state = np.zeros(p)
        self.observed_opt_state[self.boundary] = self.observed_overshoot
        self.observed_opt_state[self.interior] = self.observed_below_thresh

        _opt_linear_diag = np.ones(p)
        _opt_linear_diag[self.boundary] = self.boundary_signs
        _opt_linear_term = np.diag(_opt_linear_diag)
        _opt_offset = np.zeros(p)
        _opt_offset[self.boundary] = self.boundary_signs * self.threshold[self.boundary]

        _score_linear_term = -np.identity(p)

        self.opt_transform = (_opt_linear_term, _opt_offset)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self._setup = True

    def projection(self, opt_state):
        """
        Full projection for Langevin.
        The state here will be only the state of the optimization variables.
        for now, groups are singletons
        """
        opt_state[self.boundary] = np.maximum(opt_state[self.boundary], 0.)
        opt_state[self.interior] = np.clip(opt_state[self.interior],
                                           -self.threshold[self.interior],
                                           self.threshold[self.interior])
        return opt_state
