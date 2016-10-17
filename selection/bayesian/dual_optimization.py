import numpy as np
import regreg.api as rr
from selection.bayesian.barrier import barrier_conjugate_softmax, barrier_conjugate_log

class identity_map(rr.smooth_atom):
    def __init__(self,
                 p,
                 coef=1.,
                 offset=None,
                 quadratic=None):
        self.p = p

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):
        arg = self.apply_offset(arg)

        if mode == 'func':
            return arg
        elif mode == 'grad':
            g = np.identity(self.p)
            return g
        elif mode == 'both':
            g = np.identity(self.p)
            return arg, g
        else:
            raise ValueError('mode incorrectly specified')



class selection_probability_dual_objective(rr.smooth_atom):

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

        self.CGF_randomization = randomizer.CGF

        if self.CGF_randomization is None:
            raise ValueError('randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

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

        self.active = active

        X_E = self.X_E = X[:, active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.B_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :],np.zeros((E,p-E))])
        self.B_inactive = np.hstack([B_mE * active_signs[None, :],np.identity((p-E))])
        self.B_p = np.vstack((self.B_active,self.B_inactive))

        self.offset_active = active_signs * lagrange[active]
        self.inactive_subgrad = np.zeros(p - E)

        self.cube_bool = np.zeros(p, np.bool)
        self.cube_bool[E:] = 1
        self.dual_arg = np.linalg.inv(self.B_p).dot(np.append(self.offset_active, self.inactive_subgrad))

        self.set_parameter(mean_parameter, noise_variance)

        _barrier_star = barrier_conjugate_softmax(self.cube_bool, self.inactive_lagrange)

        self.conjugate_barrier = rr.affine_smooth(_barrier_star, np.identity(p))

        self.CGF_randomizer = rr.affine_smooth(self.CGF_randomization, -np.linalg.inv(self.B_p.T))

        self.linear_term = rr.affine_smooth(identity_map(p), self.dual_arg)

        self.total_loss = rr.smooth_sum([self.conjugate_barrier,
                                         self.CGF_randomizer,
                                         self.likelihood_loss,
                                         self.linear_term])

    def set_parameter(self, mean_parameter, noise_variance):

        mean_parameter = np.squeeze(mean_parameter)

        self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)

        self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, self._X.dot(np.linalg.inv(self.B_p.T)))

   # def objective_dual(self, param):

        #return self.conjugate_barrier.smooth_objective(self.feasible_point, mode='func')
        #return self.likelihood_loss.smooth_objective(self.feasible_point, mode='func')
        #return self.CGF_randomizer.smooth_objective(self.feasible_point, mode='func')
        #return self.linear_term.smooth_objective(self.feasible_point, mode='func')

    def smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        if mode == 'func':
            f = self.total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = self.total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")





















