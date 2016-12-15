import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm

class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange

        rr.smooth_atom.__init__(self,
                                (q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale

        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        log_cube_grad = -(np.true_divide(norm.pdf(arg_u) - norm.pdf(arg_l), cube_prob))/self.randomization_scale

        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class approximate_conditional_prob_E(rr.smooth_atom):

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
                 quadratic= None):

        self.t = t

        self.A = A

        self.target = target

        self.null_statistic = null_statistic

        E = active.sum()

        p = X.shape[1]

        self.q = p-E

        self.active = active

        self.randomization = randomizer

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

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.B_active, offset_active))

        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = 1.)

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

        objective = lambda u: self.sel_prob_smooth_objective(u, j, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, j, 'grad')

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

class approximate_conditional_density_E(rr.smooth_atom):

    def __init__(self,
                 y,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 Sigma_parameter,
                 noise_variance,
                 randomizer,
                 epsilon,
                 coef = 1.,
                 offset = None,
                 quadratic = None,
                 nstep = 10):

        (self.X, self.feasible_point, self.active, self.active_signs, self.lagrange,
         self.noise_variance, self.randomizer, self.epsilon) = (X, feasible_point, active, active_signs,
                                                                lagrange, noise_variance, randomizer, epsilon)

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

        n, p = X.shape

        nactive = self.active.sum()

        Sigma_D_T = Sigma_parameter[:, :nactive]
        Sigma_T = Sigma_parameter[:nactive, :nactive]
        Sigma_T_inv = np.linalg.inv(Sigma_T)

        X_active = X[:, active]
        X_inactive = X[:, ~active]
        B = X.T.dot(X_active)

        B_active = B[active]
        B_nactive = B[~active]

        data_active = np.hstack([-B_active, np.zeros((nactive, p - nactive))])
        data_nactive = np.hstack([-B_nactive, np.identity(p - nactive)])
        data_coef = np.vstack([data_active, data_nactive])

        self.A = (data_coef.dot(Sigma_D_T)).dot(Sigma_T_inv)

        # observed target and null statistic
        X_gen_inv = np.linalg.pinv(X_active)
        X_projection = X_active.dot(X_gen_inv)
        X_inter = (X_inactive.T).dot((np.identity(n) - X_projection))
        D_mean = np.vstack([X_gen_inv, X_inter])
        data_obs = D_mean.dot(y)
        self.target_obs = data_obs[:nactive]
        self.null_statistic = (data_coef.dot(data_obs)) -(self. A.dot(self.target_obs))

        #defining the grid on which marginal conditional densities will be evaluated
        self.grid = np.squeeze(np.round(np.linspace(-4, 10, num=141), decimals=1))
        s_obs = np.zeros(nactive)
        self.ind_obs = np.zeros(nactive, int)
        self.norm = np.zeros(nactive)
        self.h_approx = np.zeros((nactive, self.grid.shape[0]))
        for j in range(nactive):
            self.norm[j] = Sigma_T[j,j]
            s_obs[j] = np.round(self.target_obs[j], decimals=1)
            if s_obs[j] < self.grid[0]:
                s_obs[j] = self.grid[0]
            self.ind_obs[j] = int(np.where(self.grid == s_obs[j])[0])
            #print("observed index", self.ind_obs[j])
            self.h_approx[j, :] = self.approx_conditional_prob(j)
            #print("here", j)

    def approx_conditional_prob(self, j):
        h_hat = []

        for i in range(self.grid.shape[0]):
            approx = approximate_conditional_prob_E(self.X,
                                                    self.target_obs,
                                                    self.A, # the coef matrix of target
                                                    self.null_statistic, #null statistic that stays fixed
                                                    self.feasible_point,
                                                    self.active,
                                                    self.active_signs,
                                                    self.lagrange,
                                                    self.randomizer,
                                                    self.epsilon,
                                                    self.grid[i])

            h_hat.append(-(approx.minimize2(j, nstep=50)[::-1])[0])

        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.

        approx_nonnormalized = []

        for i in range(self.grid.shape[0]):
            approx_density = np.exp(-np.true_divide((self.grid[i] - mean) ** 2, 2 * (self.noise_variance * self.norm[j]))
                                    + (self.h_approx[j,:])[i])

            normalizer = normalizer + approx_density

            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer))

    def approximate_ci(self, j):

        param_grid = np.round(np.linspace(-5, 10, num=150), decimals=1)

        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):

            area_vec = self.area_normalized_density(j, param_grid[k])
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]

        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0
