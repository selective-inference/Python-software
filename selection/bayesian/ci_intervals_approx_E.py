import time
import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm


def myround(a, decimals=1):
    a_x = np.round(a, decimals=1)* 10.
    rem = np.zeros(a.shape[0], bool)
    rem[(np.remainder(a_x, 2) == 1)] = 1
    a_x[rem] = a_x[rem] + 1.
    return a_x/10.


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
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg)/(self.randomization_scale**2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg)/(self.randomization_scale**2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg>0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)
        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                        cube_prob[indicator]))/self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index])/
                                     ((prod_arg[pos_index]/arg_u[pos_index])-
                                      (1./arg_l[pos_index])))/self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] -(arg_l[neg_index]*neg_prod_arg[neg_index]))
                                    /self.randomization_scale)/(1.- neg_prod_arg[neg_index])


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
                 B_E,
                 B_mE,
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

        p = self.A.shape[0]

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

        self.feasible_point = feasible_point

        #print("feasible_point", feasible_point)

        rr.smooth_atom.__init__(self,
                                (E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=feasible_point,
                                coef=coef)

        self.coefs[:] = self.feasible_point

        self.B_active = (B_E + epsilon * np.identity(E)) * active_signs[None, :]
        self.B_inactive = B_mE * active_signs[None, :]

        self.subgrad_offset = active_signs * self.active_lagrange

        self.nonnegative_barrier = nonnegative_softmax_scaled(E)

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

    def minimize2(self, j, step=1, nstep=30, tol=1.e-6):

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
                #print("current proposal and grad", proposal, newton_step)
                if np.all(proposal > 0):
                    break
                step *= 0.5
                if count >= 40:
                    #print(proposal)
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print(current_value, proposed_value, 'minimize')
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

        self.B_E = B[active]
        self.B_mE = B[~active]

        data_active = np.hstack([-self.B_E, np.zeros((nactive, p - nactive))])
        data_nactive = np.hstack([-self.B_mE, -np.identity(p - nactive)])
        data_coef = np.vstack([data_active, data_nactive])

        self.A = (data_coef.dot(Sigma_D_T)).dot(Sigma_T_inv)

        # observed target and null statistic
        X_gen_inv = np.linalg.pinv(X_active)
        X_projection = X_active.dot(X_gen_inv)
        X_inter = (X_inactive.T).dot((np.identity(n) - X_projection))
        D_mean = np.vstack([X_gen_inv, X_inter])
        data_obs = D_mean.dot(y)
        self.target_obs = data_obs[:nactive]
        self.null_statistic = (data_coef.dot(data_obs)) -(self.A.dot(self.target_obs))

        #defining the grid on which marginal conditional densities will be evaluated
        self.grid = np.squeeze(np.round(np.linspace(-4, 8, num=121), decimals=1))
        s_obs = np.round(self.target_obs, decimals =1)
        print("observed values", s_obs)
        self.ind_obs = np.zeros(nactive, int)
        self.norm = np.zeros(nactive)
        self.h_approx = np.zeros((nactive, self.grid.shape[0]))

        for j in range(nactive):

            self.norm[j] = Sigma_T[j,j]
            if s_obs[j] < self.grid[0]:
                self.ind_obs[j] = 0
            elif s_obs[j] > np.max(self.grid):
                self.ind_obs[j] = 120
            else:
                self.ind_obs[j] = (np.where(self.grid == s_obs[j])[0])[0]
            self.h_approx[j, :] = self.approx_conditional_prob(j)

    def approx_conditional_prob(self, j):
        h_hat = []

        for i in range(self.grid.shape[0]):
            approx = approximate_conditional_prob_E(self.B_E,
                                                    self.B_mE,
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

        param_grid = np.round(np.linspace(-5, 10, num=151), decimals=1)

        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):

            area_vec = self.area_normalized_density(j, param_grid[k])
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]

        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0
