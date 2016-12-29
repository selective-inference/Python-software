import numpy as np
import regreg.api as rr
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.selection_probability_rr import cube_subproblem_scaled, cube_barrier_scaled,\
    cube_gradient_scaled, cube_hessian_scaled, cube_objective
from selection.bayesian.credible_intervals import projected_langevin


class selection_probability_objective_fs_2steps(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point, #in R^{|E|_1 + |E|_2}
                 active_1, #the active set chosen by randomized marginal screening
                 active_2, #the active set chosen by randomized lasso
                 active_signs_1, #the set of signs of active coordinates chosen by ms
                 active_signs_2, #the set of signs of active coordinates chosen by lasso
                 lagrange, #in R^p
                 threshold, #in R^p
                 mean_parameter, # in R^n
                 noise_variance,
                 randomizer,
                 epsilon, #ridge penalty for randomized lasso
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        n, p = X.shape
        E_1 = active_1.sum()
        E_2 = active_2.sum()

        self.active_1 = active_1
        self.active_2 = active_2
        self.noise_variance = noise_variance
        self.randomization = randomizer
        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = np.zeros(n + E_1 + E_2, )
        initial[n:] = feasible_point

        rr.smooth_atom.__init__(self,
                                (n + E_1 + E_2,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial
        nonnegative = nonnegative_softmax(E_1 + E_2)
        opt_vars = np.zeros(n + E_1 + E_2, bool)
        opt_vars[n:] = 1

        self._opt_selector = rr.selector(opt_vars, (n + E_1 + E_2,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + E_1 + E_2,))

        X_E_1 = X[:, active_1]
        X_mE_1 = X[:, ~active_1]

        X_E_2 = X[:, active_2]
        B = X.T.dot(X_E_2)

        B_E = B[active_2]
        B_mE = B[~active_2]

        self.A_active_2 = np.hstack([-X[:, active_2].T, (B_E + epsilon * np.identity(E_2)) * active_signs_2[None, :]])
        self.A_inactive_2 = np.hstack([-X[:, ~active_2].T, (B_mE * active_signs_2[None, :])])

        self.offset_active_2 = active_signs_2 * lagrange[active_2]

        self.active_conj_loss_2 = rr.affine_smooth(self.active_conjugate,
                                                   rr.affine_transform(self.A_active_2, self.offset_active_2))

        cube_obj_2 = cube_objective(self.inactive_conjugate, lagrange[~active_2], nstep=nstep)

        self.cube_loss_2 = rr.affine_smooth(cube_obj_2, self.A_inactive_2)








