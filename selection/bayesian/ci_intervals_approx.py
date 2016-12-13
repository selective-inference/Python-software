import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled, cube_subproblem_scaled, \
    cube_barrier_scaled, cube_gradient_scaled, cube_hessian_scaled, cube_objective

class approximate_conditional_prob(rr.smooth_atom):

    def __init__(self,
                 X,
                 target,
                 A, # the coef matrix of target
                 null_statistic, #null statistic that stays fixed
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 randomizer,
                 epsilon,
                 t, #point at which density is to computed
                 coef = 1.,
                 offset= None,
                 quadratic= None,
                 nstep = 10):

        self.t = t

        self.A = A

        self.target = target

        self.null_statistic = null_statistic

        E = active.sum()

        self.active = active

        self.randomization = randomizer

        self.nstep = nstep

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]
        self.active_lagrange = lagrange[active]

        #here, feasible point is in E dimensions
        initial = feasible_point

        self.feasible_point = feasible_point

        rr.smooth_atom.__init__(self,
                                (E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        nonnegative = nonnegative_softmax_scaled(E)

        X_E = self.X_E = X[:, active]
        self.X_inactive = X[:, ~active]

        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.B_active = (B_E + epsilon * np.identity(E)) * active_signs[None, :]
        self.B_inactive = B_mE * active_signs[None, :]

        self.subgrad_offset = active_signs * self.active_lagrange

        opt_vars = np.zeros(E, bool)
        opt_vars[:E] = 1

        self._opt_selector = rr.selector(opt_vars, (E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)

        self.E = E

    def sel_prob_smooth_objective(self, param, j, mode='both', check_feasibility=False):

        param = self.apply_offset(param)
        index = np.zeros(self.E, bool)
        index[j] = 1
        data = np.squeeze(self.t * self.A[:, index]) + self.A[:, ~index].dot(self.target[~index])

        offset_active = self.subgrad_offset + self.null_statistic[:self.E] + data[:self.E]

        offset_inactive = self.null_statistic[self.E:] + data[self.E:]

        print("shape of offsets", np.shape(offset_active), np.shape(offset_inactive))

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                                 rr.affine_transform(self.B_active, offset_active))

        cube_obj = cube_objective(self.inactive_conjugate, self.inactive_lagrange, nstep=self.nstep)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.B_inactive, offset_inactive))

        total_loss = rr.smooth_sum([active_conj_loss,
                                    cube_loss,
                                    self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, j, step=1, nstep=30, tol=1.e-8):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(j, u, 'func')
        grad = lambda u: self.smooth_objective(j, u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current)

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.all(proposal > 0):
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

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        # print('iter', itercount)
        value = objective(current)
        return current, value

