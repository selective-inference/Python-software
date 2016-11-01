import numpy as np
import regreg.api as rr
from selection.bayesian.barrier import barrier_conjugate_softmax, barrier_conjugate_softmax_scaled,\
    barrier_conjugate_log, cube_barrier_softmax_coord, barrier_conjugate_softmax_scaled_rr


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

        self.coefs[:] = feasible_point

        mean_parameter = np.squeeze(mean_parameter)

        self.active = active

        X_E = self.X_E = X[:, active]
        self.X_permute = np.hstack([self.X_E, self._X[:, ~active]])
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.active_slice = np.zeros_like(active, np.bool)
        self.active_slice[:active.sum()] = True

        self.B_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :],np.zeros((E,p-E))])
        self.B_inactive = np.hstack([B_mE * active_signs[None, :], np.identity((p-E))])
        self.B_p = np.vstack((self.B_active,self.B_inactive))

        self.B_p_inv = np.linalg.inv(self.B_p.T)

        self.offset_active = active_signs * lagrange[active]
        self.inactive_subgrad = np.zeros(p - E)

        self.cube_bool = np.zeros(p, np.bool)

        self.cube_bool[E:] = 1

        self.dual_arg = self.B_p_inv.dot(np.append(self.offset_active, self.inactive_subgrad))

        self._opt_selector = rr.selector(~self.cube_bool, (p,))

        self.set_parameter(mean_parameter, noise_variance)

        _barrier_star = barrier_conjugate_softmax_scaled_rr(self.cube_bool, self.inactive_lagrange)

        #_barrier_star = barrier_conjugate_log(self.cube_bool, self.inactive_lagrange)

        self.conjugate_barrier = rr.affine_smooth(_barrier_star, np.identity(p))

        self.CGF_randomizer = rr.affine_smooth(self.CGF_randomization, -self.B_p_inv)

        #self._linear_term = linear_map(p, self.dual_arg)

        self.constant = np.true_divide(mean_parameter.dot(mean_parameter), 2*noise_variance)

        self.linear_term = rr.identity_quadratic(0, 0, self.dual_arg, -self.constant)

        self.total_loss = rr.smooth_sum([self.conjugate_barrier,
                                         self.CGF_randomizer,
                                         self.likelihood_loss])

        self.total_loss.quadratic = self.linear_term

    def set_parameter(self, mean_parameter, noise_variance):

        mean_parameter = np.squeeze(mean_parameter)

        self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)

        self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, self.X_permute.dot(self.B_p_inv))

    def _smooth_objective(self, param, mode='both', check_feasibility=False):

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

    def minimize(self, initial=None, min_its=10, max_its=50, tol=1.e-10):

        nonpos_con = self._opt_selector.output_shape[0]
        constraint = rr.separable(self.shape,
                                  [rr.nonpositive((nonpos_con,), offset=-1.e-12 * np.ones(nonpos_con))],
                                  [self._opt_selector.index_obj])

        problem = rr.separable_problem.fromatom(constraint, self.total_loss)
        problem.coefs[:] = self.coefs
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        self.coefs[:] = soln
        value = problem.objective(soln)
        return soln, value

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        n, p = self._X.shape

        current = self.feasible_point
        current_value = np.inf

        objective = lambda u: self.total_loss.objective(u)
        grad = lambda u: self.total_loss.smooth_objective(u, 'grad') + self.dual_arg

        for itercount in range(nstep):
            newton_step = grad(current) * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.all(proposal[self.active_slice] < 0):
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

        #print('iter', itercount)
        value = objective(current)
        return current, value























