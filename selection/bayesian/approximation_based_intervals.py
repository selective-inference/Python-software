import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled, cube_subproblem_scaled, \
    cube_barrier_scaled, cube_gradient_scaled, cube_hessian_scaled, cube_objective

#class should return approximate probability of (\beta_E,u_{-E}) in K conditional on s:
class approximate_conditional_sel_prob(rr.smooth_atom):

    def __init__(self,
                 y,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 randomizer,
                 epsilon,
                 j, #index of interest amongst active variables
                 s, #point at which density is to computed
                 coef = 1.,
                 offset= None,
                 quadratic= None,
                 nstep = 10):

        n, p = X.shape

        E = active.sum()

        self.y = y

        self.active = active

        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

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

        self.subgrad_offset = active_signs * lagrange[active]

        self.j = j

        self.s = s

        eta = np.linalg.pinv(self.X_E)[self.j, :]
        c = np.true_divide(eta, np.linalg.norm(eta) ** 2)

        fixed_part = (np.identity(n) - np.outer(c, eta)).dot(self.y)
        self.offset_active = self.subgrad_offset - self.X_E.T.dot(fixed_part) - self.s * (self.X_E.T.dot(c))
        self.offset_inactive = - self.X_inactive.T.dot(fixed_part) - self.s * (self.X_inactive.T.dot(c))

        opt_vars = np.zeros(E, bool)
        opt_vars[:E] = 1

        self._opt_selector = rr.selector(opt_vars, (E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)

        self.active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                                 rr.affine_transform(self.B_active, self.offset_active))

        cube_obj = cube_objective(self.inactive_conjugate,
                                  lagrange[~active],
                                  nstep=nstep)

        self.cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.B_inactive, self.offset_inactive))

        self.total_loss = rr.smooth_sum([self.active_conj_loss,
                                         self.cube_loss,
                                         self.nonnegative_barrier])

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

    def minimize(self, initial=None, min_its=100, max_its=500, tol=1.e-10):

        nonneg_con = self._opt_selector.output_shape[0]
        constraint = rr.separable(self.shape,
                                  [rr.nonnegative((nonneg_con,), offset=1.e-12 * np.ones(nonneg_con))],
                                  [self._opt_selector.index_obj])

        problem = rr.separable_problem.fromatom(constraint, self.total_loss)
        problem.coefs[:] = self.coefs
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        self.coefs[:] = soln
        value = problem.objective(soln)
        return soln, value

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

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


class approximate_conditional_density(rr.smooth_atom):

    def __init__(self,
                 y,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 noise_variance,
                 randomizer,
                 epsilon,
                 j,  # index of interest amongst active variables
                 coef = 1.,
                 offset = None,
                 quadratic = None,
                 nstep = 10):

        (self.y, self.X, self.feasible_point, self.active, self.active_signs, self.lagrange,
         self.noise_variance, self.randomizer, self.epsilon, self.j) = (y, X, feasible_point, active, active_signs,
                                                                       lagrange, noise_variance, randomizer, epsilon, j)
        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)


        self.grid = np.squeeze(np.round(np.linspace(-4, 10, num = 141), decimals=1))

        self.contrast = np.linalg.pinv(self.X[:,active])[self.j, :]

        self.norm = np.linalg.norm(self.contrast)**2

        s_obs = np.round(np.squeeze(self.contrast.dot(self.y)), decimals=1)

        if s_obs>=self.grid[0]:
            s_obs = s_obs
        else:
            s_obs = self.grid[0]

        self.ind_obs =  np.where(self.grid == s_obs)[0]

        #print self.grid, s_obs, self.ind_obs

        self.h_approx = self.approx_conditional()


    def approx_conditional(self):

        h_hat = []

        for i in range(self.grid.shape[0]):

            approx = approximate_conditional_sel_prob(self.y,
                                                      self.X,
                                                      self.feasible_point,
                                                      self.active,
                                                      self.active_signs,
                                                      self.lagrange,
                                                      self.randomizer,
                                                      self.epsilon,
                                                      self.j, #index of interest amongst active variables
                                                      self.grid[i])

            h_hat.append(-(approx.minimize2(nstep = 80)[::-1])[0])

            #print i

        return np.array(h_hat)

    def normalized_density(self, truth):

        normalizer = 0.

        approx_nonnormalized = []

        for i in range(self.grid.shape[0]):

            approx_density = np.exp(-np.true_divide((self.grid[i]-truth)**2, 2*(self.noise_variance * self.norm))
                                    + self.h_approx[i])

            normalizer = normalizer + approx_density

            approx_nonnormalized.append(approx_density)

            #print i, approx_density

        return np.array(approx_nonnormalized/normalizer)

    def approximate_ci(self):

        param_grid = np.round(np.linspace(0, 10, num= 100), decimals=1)

        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):

            area_vec = np.cumsum(self.normalized_density(param_grid[k]))

            area[k] = area_vec[self.ind_obs]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]

        if region.size>0:
            return np.nanmin(region), np.nanmax(region)

        else:
            return 0,0















































