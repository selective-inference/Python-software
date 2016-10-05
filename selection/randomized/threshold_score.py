import numpy as np
import regreg.api as rr

from .M_estimator import M_estimator, restricted_Mest

class threshold_score(M_estimator):

    def __init__(self, loss, threshold, randomization, active_groups, inactive_groups, beta_active):
        """
        penalty is a group_lasso object that assigns weights to groups
        """

        (self.loss,
         self.active_groups,
         self.inactive_groups,
         self.beta_active,
         self.randomization) = (loss,
                                active_groups,
                                inactive_groups,
                                beta_active,
                                randomization)

    def solve(self):

        (loss,
         active_groups,
         inactive_groups,
         beta_active,
         randomization) = (self.loss,
                           self.active_groups,
                           self.inactive_groups,
                           self.beta_active,
                           self.randomization)

        self.randomize()
         
        beta_full = np.zeros(self.loss.shape)
        beta_full[active_groups] = beta_active

        inactive_score = self.loss.smooth_objective(beta_full, 'grad')[inactive_groups]
 
        # find the current active group, i.e. 
        # subset of inactive_groups that pass the threshold

        # TODO: make this test use group LASSO 

        self.active = np.fabs(inactive_score + self._random_term.linear) > threshold
        self.active_signs = np.sign(inactive_score + self._random_term.linear)
        self.inactive = ~inactive

        self.observed_overshoot = self.active_signs * (inactive_score[self.active] - threshold)
        self.observed_underthreshold = inactive_score[self.inactive]
        self.observed_score = inactive_score

        self.selection_variable = {'boundary_set':self.active,
                                   'boundary_signs':self.active_signs}
        

    def setup_sampler(self):

        self.observed_opt_state = np.hstack([self.observed_underthreshold,
                                             self.observed_overshoot])

        p = self.inactive.sum() # shorthand
        _opt_linear_term = np.zeros((p, 1 + self.observed_subgradients.shape[0]))
        _opt_linear_term[:,:self.observed_subgradients.shape[0]] = self.losing_padding_map
        _opt_linear_term[:,-1] = self.maximizing_subgrad

        _score_linear_term = np.identity(p)

        self.opt_transform = (_opt_linear_term, np.zeros(_opt_linear_term.shape[0]))
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

    def projection(self, opt_state):
        """
        Full projection for Langevin.

        The state here will be only the state of the optimization variables.
        """
        return self.group_lasso_dual_epigraph.cone_prox(opt_state)

