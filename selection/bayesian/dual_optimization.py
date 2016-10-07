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
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.X=X
        n, p = X.shape
        E = active.sum()

        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomization

        self.CGF_randomization = randomization.CGF

        if self.CGF_randomization is None:
            raise ValueError(
                'randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        X_E = self.X_E = X[:,active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :],np.zeros((E,p-E))])
        self.A_inactive = np.hstack([B_mE * active_signs[None, :],np.identity((p-E))])
        self.A=np.vstack((self.A_active,self.A_inactive))
        self.dual_arg = np.zeros(p)
        self.dual_arg[:E] = -active_signs * lagrange[active]

        initial=feasible_point

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.cube_bool = np.zeros(p, np.bool)
        self.cube_bool[E:] = 1

        self.set_parameter(mean_parameter, noise_variance)

        self.coefs[:] = initial

    def set_parameter(self, mean_parameter, noise_variance):

        self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)
        #self.likelihood_loss.quadratic = rr.identity_quadratic(0, 0, 0,
                                                             # -0.5 * (mean_parameter ** 2).sum() / noise_variance)

        self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, self.X)

    def smooth_objective(self, dual, mode='both', check_feasibility=False):

        dual = self.apply_offset(dual)

        _barrier_star = barrier_conjugate(self.cube_bool,self.inactive_lagrange)

        composition_barrier = rr.affine_smooth(_barrier_star, self.A.T)

        CGF_rand_value, CGF_rand_grad = self.CGF_randomization

        if mode == 'func':
            f_rand_cgf = CGF_rand_value(dual)
            f_data_cgf = self.likelihood_loss.smooth_objective(dual, 'func')
            f_barrier_conj=composition_barrier.smooth_objective(dual, 'func')
            f = self.scale(f_rand_cgf + f_data_cgf + f_barrier_conj-(dual.T.dot(self.dual_arg)))
            # print(f, f_nonneg, f_like, f_active_conj, conjugate_value_i, 'value')
            return f

        elif mode == 'grad':
            g_rand_cgf = CGF_rand_grad(dual)
            g_data_cgf = self.likelihood_loss.smooth_objective(dual, 'grad')
            g_barrier_conj = composition_barrier.smooth_objective(dual, 'grad')
            g = self.scale(g_rand_cgf + g_data_cgf + g_barrier_conj-self.dual_arg)
            # print(g, 'grad')
            return g

        elif mode == 'both':
            f_rand_cgf, g_rand_cgf = self.CGF_randomization(dual)
            f_data_cgf, g_data_cgf = self.likelihood_loss.smooth_objective(dual, 'both')
            f_barrier_conj, g_barrier_conj = composition_barrier.smooth_objective(dual, 'both')
            f = self.scale(f_rand_cgf + f_data_cgf + f_barrier_conj-(dual.T.dot(self.dual_arg)))
            g = self.scale(g_rand_cgf + g_data_cgf + g_barrier_conj-self.dual_arg)
            # print(f, f_nonneg, f_like, f_active_conj, conjugate_value_i, 'value')
            return f, g
        else:
            raise ValueError("mode incorrectly specified")

    def minimize(self, initial=None, step=1, nstep=30):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')

        for itercount in range(nstep):
            newton_step = self.smooth_objective(current, 'grad') * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.isfinite(objective(proposal)):
                    break
                step *= 0.5
                if count >= 40:
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                # print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < 1.e-6 * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        value = objective(current)
        return current, value




















