"""
Projected Langevin sampler of `http://arxiv.org/abs/1507.02564`_
"""
from __future__ import print_function

import numpy as np
from scipy.stats import norm as ndist

class projected_langevin(object):

    def __init__(self, 
                 initial_condition,
                 gradient_map,
                 projection_map,
                 stepsize):

        (self.state,
         self.gradient_map,
         self.projection_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           projection_map,
                           stepsize)
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0,scale=1)

    def __iter__(self):
        return self

    def next(self):
        nattempt = 0
        while True:
            
            proj_arg = (self.state
                        + 0.5 * self.stepsize * self.gradient_map(self.state)
                        + self._noise.rvs(self._shape) * self._sqrt_step)
            candidate = self.projection_map(proj_arg)
            if not np.all(np.isfinite(self.gradient_map(candidate))):
                nattempt += 1
                self._sqrt_step *= 0.8
                if nattempt >= 10:
                    raise ValueError('unable to find feasible step')
            else:
                self.state[:] = candidate
                break
