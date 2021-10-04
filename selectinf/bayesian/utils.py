import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.stats import norm as ndist

class langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 stepsize,
                 proposal_scale):

        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           stepsize)
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0,scale=1)
        self.sample = np.copy(initial_condition)

        self.proposal_scale = proposal_scale
        self.proposal_sqrt = fractional_matrix_power(self.proposal_scale, 0.5)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        while True:

            gradient_posterior, draw, _ = self.gradient_map(self.state)

            candidate = (self.state + self.stepsize * self.proposal_scale.dot(gradient_posterior)
                         + np.sqrt(2.) * (self.proposal_sqrt.dot(self._noise.rvs(self._shape))) * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate)[0])):
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                self.sample[:] = draw
                #print(" next sample ", self.state[:], self.sample[:])
                break

            return self.sample

def langevin_sampler(posterior,
                     nsample=2000,
                     nburnin=100,
                     step_frac=0.3,
                     start=None):

    if start is None:
        start = posterior.initialize_sampler(posterior.initial_estimate)

    state = np.append(start, np.ones(posterior.target_size))
    stepsize = 1. / (step_frac * (2 * posterior.target_size))
    proposal_scale = np.identity(int(2 * posterior.target_size))
    sampler = langevin(state, posterior.gradient_log_likelihood, stepsize, proposal_scale)

    samples = np.zeros((nsample, 2 * posterior.target_size))

    for i, sample in enumerate(sampler):
        samples[i, :] = sampler.sample.copy()
        print(" next sample ", i, samples[i, :])
        if i == nsample - 1:
            break

    return samples[nburnin:, :]