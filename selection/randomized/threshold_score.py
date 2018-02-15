import functools

import numpy as np
import regreg.api as rr

from .query import query, optimization_sampler
from .base import restricted_estimator

class threshold_score(query):

    """

    Randomly threshold the score of a linear 
    model.

    """

    def __init__(self, 
                 loss, 
                 threshold, 
                 randomization, 
                 active, 
                 candidate, 
                 beta_active=None,
                 solve_args={'min_its': 50, 'tol': 1.e-10}):
        """

        Parameters
        ----------

        loss : regreg.smooth.smooth_atom
            Loss whose score (gradient) will be thresholded.

        threshold_value : [float, sequence]
            Thresholding for each feature. If 1d defaults
            it is treated as a multiple of np.ones.

        randomization : selection.randomized.randomization.randomization
            Instance of a randomizer.

        active : np.bool
            Loss is first partially minimized over the active coordinates.
            May be all zeros.

        candidate : np.bool
            Candidate coordinates for thresholding.
        
        beta_active : np.float (optional)
            If supplied this is taken as solution 
            of partial minimization.

        solve_args : dict (optional)
            Arguments passed in solving the partial minimization.
        """

        query.__init__(self, randomization)

        # threshold could be a vector size candidate

        active_bool = np.zeros(loss.shape, np.bool)
        active_bool[active] = 1
        active = active_bool

        if np.array(threshold).shape in [(), (1,)]:
            threshold = np.ones(candidate.sum()) * threshold

        self.epsilon = 0.  # for randomized loss

        (self.loss,
         self.threshold,
         self.active,
         self.candidate,
         self.beta_active,
         self.randomization,
         self.solve_args) = (loss,
                             threshold,
                             active,
                             candidate,
                             beta_active,
                             randomization,
                             solve_args)

    def solve(self, nboot=2000):

        (loss,
         threshold,
         active,
         candidate,
         beta_active,
         randomization) = (self.loss,
                           self.threshold,
                           self.active,
                           self.candidate,
                           self.beta_active,
                           self.randomization)

        self._marginalize_subgradient = True # need to find a better place to set this...

        if beta_active is None:
            beta_active = self.beta_active = restricted_estimator(self.loss, active, solve_args=self.solve_args)

        self.randomize()

        beta_full = np.zeros(self.loss.shape)
        beta_full[active] = beta_active
        self._beta_full = beta_full

        candidate_score = self.loss.smooth_objective(beta_full, 'grad')[candidate]
        randomized_score = candidate_score + randomization.sample()

        # find the current active group, i.e.
        # subset of candidate that pass the threshold

        # TODO: make this test use group LASSO

        self.boundary = np.fabs(randomized_score) > threshold

        self.interior = ~self.boundary

        self.observed_internal_state = self.observed_score_state = candidate_score

        active_signs = np.sign(randomized_score[self.boundary])
        self.selection_variable = {'boundary_set': self.boundary,
                                   'active_signs': active_signs}

        self._solved = True

        self.nboot = nboot
        self.ndim = self.loss.shape[0]

        # must set observed_opt_state, opt_transform and score_transform

        p = self.boundary.shape[0]  # shorthand
        self.num_opt_var = 0
        opt_transform = np.identity(p)
        opt_transform = np.vstack([opt_transform[self.boundary], opt_transform[self.interior]])
        opt_offset = np.hstack([active_signs * threshold[self.boundary], 
                                np.zeros(self.interior.sum())])
        self.opt_transform = (opt_transform, opt_offset)
        self.observed_opt_state = np.hstack([active_signs * threshold[self.boundary], 
                                             randomized_score[self.interior]])
        _score_linear_term = -np.identity(p)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self._setup = True

    def get_sampler(self):

        if not hasattr(self, "_sampler"):

            def log_density(boundary, 
                            threshold,
                            _density,
                            _cdf,
                            score_state, 
                            opt_state):
                """
                marginalizing over the sub-gradient
                """

                logdens = 0
                weights = np.zeros_like(boundary, np.float)

                logdens += np.log(1 - _cdf(threshold[boundary] - score_state[:, boundary]) + 
                                  _cdf(-threshold[boundary] - score_state[:, boundary])).sum()
                logdens += np.log(_cdf(threshold[~boundary] - score_state[:, ~boundary]) - 
                                   _cdf(-threshold[~boundary] - score_state[:, ~boundary])).sum()
                return logdens
            

            log_density = functools.partial(log_density,
                                            self.boundary,
                                            self.threshold,
                                            self.randomization._density,
                                            self.randomization._cdf)

            # the gradient and projection are used for 
            # Langevin sampling of opt variables
            # but this view has no opt variables

            grad_log_density = None
            projection = None

            self._sampler = optimization_sampler(np.zeros(()), # nothing to sample
                                                 self.observed_score_state,
                                                 self.score_transform,
                                                 self.opt_transform,
                                                 projection,
                                                 grad_log_density,
                                                 log_density)
        return self._sampler

    sampler = property(get_sampler, query.set_sampler)

    def setup_sampler(self):
        pass


