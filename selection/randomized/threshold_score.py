import functools

import numpy as np
import regreg.api as rr

from .query import query, optimization_sampler
from .reconstruction import reconstruct_full_from_internal
from .M_estimator import restricted_Mest

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
            beta_active = self.beta_active = restricted_Mest(self.loss, active, solve_args=self.solve_args)

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

        self.observed_internal_state = candidate_score

        self.selection_variable = {'boundary_set': self.boundary}

        self._solved = True

        self.nboot = nboot
        self.ndim = self.loss.shape[0]

        # must set observed_opt_state, opt_transform and score_transform

        p = self.boundary.shape[0]  # shorthand
        self.num_opt_var = 0
        self.opt_transform = (np.array([], np.float), np.zeros(p, np.float))
        self.observed_opt_state = np.array([])
        _score_linear_term = -np.identity(p)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self._setup = True

    def get_sampler(self):

        if not hasattr(self, "_sampler"):

            def grad_log_density(boundary, 
                                 opt_transform,
                                 score_transform,
                                 threshold,
                                 _density,
                                 _cdf,
                                 internal_state, 
                                 opt_state):
                """
                marginalizing over the sub-gradient
                """

                full_state = reconstruct_full_from_internal(opt_transform, score_transform, internal_state, opt_state)

                weights = np.zeros_like(boundary, np.float)

                weights[boundary] = ((_density(threshold[boundary] - full_state[boundary])
                                           - _density(-threshold[boundary] - full_state[boundary])) /
                                          (1 - _cdf(threshold[boundary] - full_state[boundary]) + 
                                           _cdf(-threshold[boundary] - full_state[boundary])))


                weights[~boundary] = ((-_density(threshold[~boundary] - 
                                                 full_state[~boundary]) + 
                                             _density(-threshold[~boundary] - full_state[~boundary])) /
                                           (_cdf(threshold[~boundary] - full_state[~boundary]) - 
                                            _cdf(-threshold[~boundary] - full_state[~boundary])))

                opt_linear = opt_transform[0]
                return opt_linear.T.dot(weights) ## tested

            grad_log_density = functools.partial(grad_log_density,
                                                 self.boundary,
                                                 self.opt_transform,
                                                 self.score_transform,
                                                 self.threshold,
                                                 self.randomization._density,
                                                 self.randomization._cdf)

            def log_density(boundary, 
                            opt_transform,
                            score_transform,
                            threshold,
                            _density,
                            _cdf,
                            internal_state, 
                            opt_state):
                """
                marginalizing over the sub-gradient
                """

                full_state = reconstruct_full_from_internal(opt_transform, score_transform, internal_state, opt_state)
                logdens = 0
                weights = np.zeros_like(boundary, np.float)

                logdens += np.log(1 - _cdf(threshold[boundary] - full_state[boundary]) + 
                                  _cdf(-threshold[boundary] - full_state[boundary]))
                logdens += np.log(_cdf(threshold[~boundary] - full_state[~boundary]) - 
                                   _cdf(-threshold[~boundary] - full_state[~boundary]))
                return logdens
            

            log_density = functools.partial(log_density,
                                            self.boundary,
                                            self.opt_transform,
                                            self.score_transform,
                                            self.threshold,
                                            self.randomization._density,
                                            self.randomization._cdf)
            projection = lambda opt: opt

            self._sampler = optimization_sampler(self.observed_opt_state,
                                                 self.observed_internal_state.copy(),
                                                 self.score_transform,
                                                 self.opt_transform,
                                                 projection,
                                                 grad_log_density,
                                                 log_density)
        return self._sampler

    sampler = property(get_sampler, query.set_sampler)

    def setup_sampler(self):
        pass


