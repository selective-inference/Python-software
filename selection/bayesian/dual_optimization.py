import numpy as np
import regreg.api as rr
from selection.bayesian.barrier import barrier_conjugate

class dual_selection_probability(rr.smooth_atom):

    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,
                 noise_variance,
                 randomization,
                 coef=1.,
                 epsilon=0.,
                 offset=None,
                 quadratic=None):

        n, p = X.shape
        E = active.sum()

        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomization

        self.CGF_perturbation = randomization.CGF

        if self.CGF_perturbation is None:
            raise ValueError(
                'randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        X_E = self.X_E = X[:, active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :],np.zeros((E,p-E))])
        self.A_inactive = np.hstack([B_mE,np.identity((p-E))*lagrange[~active]])
        self.A=np.vstack(self.A_active,self.A_inactive)
        self.offset = np.zeros(p)
        self.offset[:E] = -active_signs * lagrange[active]

        initial=feasible_point

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        cube_bool = np.zeros(p, np.bool)
        cube_bool[E:] = 1

        self.set_parameter(mean_parameter, noise_variance)

        def set_parameter(self, mean_parameter, noise_variance):

            self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)
            self.likelihood_loss.quadratic = rr.identity_quadratic(0, 0, 0,
                                                              -0.5 * (mean_parameter ** 2).sum() / noise_variance)

            self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, X)

        def smooth_objective(self, dual, mode='both', check_feasibility=False):

            dual = self.apply_offset(dual)

            _barrier_star = barrier_conjugate(cube_bool,lagrange[~active])

            composition_barrier = rr.affine_smooth(_barrier_star, self.A.T)

            CGF_pert_value, CGF_pert_grad = self.CGF_perturbation

            if mode == 'func':
                f_pert_cgf = CGF_pert_value(dual)
                f_data_cgf = self.likelihood_loss.smooth_objective(dual, 'func')
                f_barrier_conj=composition_barrier.smooth_objective(dual, 'func')
                f = self.scale(f_pert_cgf + f_data_cgf + f_barrier_conj-(dual.T.dot(self.offset)))
                # print(f, f_nonneg, f_like, f_active_conj, conjugate_value_i, 'value')
                return f

            elif mode == 'grad':
                g_pert_cgf = CGF_pert_grad(dual)
                g_data_cgf = self.likelihood_loss.smooth_objective(dual, 'grad')
                g_barrier_conj = composition_barrier.smooth_objective(dual, 'grad')
                g = self.scale(g_pert_cgf + g_data_cgf + g_barrier_conj-self.offset)
                # print(g, 'grad')
                return g

            elif mode == 'both':
                f_pert_cgf, g_pert_cgf = self.CGF_perturbation(dual)
                f_data_cgf, g_data_cgf = self.likelihood_loss.smooth_objective(dual, 'both')
                f_barrier_conj, g_barrier_conj = composition_barrier.smooth_objective(dual, 'both')
                f = self.scale(f_pert_cgf + f_data_cgf + f_barrier_conj-(dual.T.dot(self.offset)))
                g = self.scale(g_pert_cgf + g_data_cgf + g_barrier_conj-self.offset)
                # print(f, f_nonneg, f_like, f_active_conj, conjugate_value_i, 'value')
                return f, g
            else:
                raise ValueError("mode incorrectly specified")



















