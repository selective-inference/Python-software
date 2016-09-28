##############langevin random walk to sample from posterior
class langevin_rw(object):
    def __init__(self,initial_state,gradient_map,stepsize):
        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_state),gradient_map,stepsize)
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0, scale=1)

    def __iter__(self):
        return self

    def next(self):
        while True:
            candidate = (self.state
                        + 0.5 * self.stepsize * self.gradient_map(self.state)
                        + self._noise.rvs(self._shape) * self._sqrt_step)
            if not np.all(np.isfinite(self.gradient_map(candidate))):
                print candidate, self._sqrt_step
                self._sqrt_step *= 0.8
            else:
                self.state[:] = candidate
                break