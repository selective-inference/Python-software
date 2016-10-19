import numpy as np
import regreg.api as rr
from selection.algorithms.softmax import nonnegative_softmax

#class should return approximate probability of (\beta_E,u_{-E}) in K conditional on s:
class approximate_density(rr.smooth_atom):

    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,  # in R^n
                 noise_variance,
                 randomizer,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        n, p = X.shape
        E = active.sum()
        self._X = X
        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        initial = feasible_point

        self.feasible_point = feasible_point

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        #self.coefs[:] = initial

        nonnegative = nonnegative_softmax(E)
        self.cube_bool = np.zeros(p, bool)
        self.cube_bool[E:] = 1

        self._opt_selector = rr.selector(~self.cube_bool, (p,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)

        X_E = self.X_E = X[:, active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.B_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :], np.zeros((E, p - E))])
        self.B_inactive = np.hstack([B_mE * active_signs[None, :], np.identity((p - E))])
        self.B_p = np.vstack((self.B_active, self.B_inactive))

        self.offset_active = (active_signs * lagrange[active])-(s*(self.X_E.T.dot(self.c)))- (self.X_E.T.dot(self.Z))
        self.offset_inactive = -()

























