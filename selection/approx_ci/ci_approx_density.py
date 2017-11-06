from __future__ import print_function
from math import log
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

class nonnegative_softmax_scaled(rr.smooth_atom):
    """
    The nonnegative softmax objective
    .. math::
         \mu \mapsto
         \sum_{i=1}^{m} \log \left(1 +
         \frac{1}{\mu_i} \right)
    """

    objective_template = r"""\text{nonneg_softmax}\left(%(var)s\right)"""

    def __init__(self,
                 shape,
                 barrier_scale=1.,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                                shape,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        # a feasible point
        self.coefs[:] = np.ones(shape)
        self.barrier_scale = barrier_scale

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        slack = self.apply_offset(mean_param)

        if mode in ['both', 'func']:
            if np.all(slack > 0):
                f = self.scale(np.log((slack + self.barrier_scale) / slack).sum())
            else:
                f = np.inf
        if mode in ['both', 'grad']:
            g = self.scale(1. / (slack + self.barrier_scale) - 1. / slack)

        if mode == 'both':
            return f, g
        elif mode == 'grad':
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")

class neg_log_cube_probability_laplace(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.b = randomization_scale
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

        arg_u = (arg + self.lagrange)/self.b
        arg_l = (arg - self.lagrange)/self.b
        scaled_lagrange = (2* self.lagrange)/self.b

        ind_arg_1 = np.zeros(self.q, bool)
        ind_arg_1[(arg_u <0.)] = 1
        ind_arg_2 = np.zeros(self.q, bool)
        ind_arg_2[(arg_l >0.)] = 1
        ind_arg_3 = np.logical_and(~ind_arg_1, ~ind_arg_2)
        cube_prob = np.zeros(self.q)
        cube_prob[ind_arg_1] = np.exp(arg_u[ind_arg_1])/2. - np.exp(arg_l[ind_arg_1])/2.
        cube_prob[ind_arg_2] = -np.exp(-arg_u[ind_arg_2])/2. + np.exp(-arg_l[ind_arg_2])/2.
        cube_prob[ind_arg_3] = 1- np.exp(-arg_u[ind_arg_3])/2. - np.exp(arg_l[ind_arg_3])/2.
        neg_log_cube_prob = -np.log(cube_prob).sum()

        log_cube_grad = np.zeros(self.q)
        log_cube_grad[ind_arg_1] = 1./self.b
        log_cube_grad[ind_arg_2] = np.true_divide((np.exp(-scaled_lagrange[ind_arg_2])+ 1.)/self.b,
                                                  np.exp(-scaled_lagrange[ind_arg_2])-1.)
        num_cube_grad = np.true_divide(np.exp(-scaled_lagrange[ind_arg_3]), 2 * self.b) - \
                        np.true_divide(np.exp((2* arg_l[ind_arg_3])), 2 * self.b)
        den_cube_grad = np.exp(arg_l[ind_arg_3]) - np.exp(-scaled_lagrange[ind_arg_3])/2. - \
                        np.exp(2* arg_l[ind_arg_3])/2.
        log_cube_grad[ind_arg_3] = np.true_divide(num_cube_grad,den_cube_grad)
        neg_log_cube_grad = -log_cube_grad

        if mode == 'func':
            return self.scale(neg_log_cube_prob)
        elif mode == 'grad':
            return self.scale(neg_log_cube_grad)
        elif mode == 'both':
            return self.scale(neg_log_cube_prob), self.scale(neg_log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")

class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q,  # equals p - E in our case
                 lagrange,
                 randomization_scale=1.,  # equals the randomization variance in our case
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

        arg_u = (arg + self.lagrange) / self.randomization_scale
        arg_l = (arg - self.lagrange) / self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg) / (self.randomization_scale ** 2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg) / (self.randomization_scale ** 2))
        cube_prob = normal.cdf(arg_u) - normal.cdf(arg_l)

        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg > 0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)

        log_cube_prob = np.zeros(self.q)
        log_cube_prob[indicator] = -np.log(cube_prob)[indicator]

        random_var = self.randomization_scale ** 2
        log_cube_prob[neg_index] = (arg[neg_index] ** 2. / (2. * random_var)) + (
        arg[neg_index] * self.lagrange[neg_index] / random_var) + \
                                   (self.lagrange[neg_index] ** 2. / (2. * random_var)) \
                                   - np.log(
            1. / np.abs(arg_u[neg_index]) - neg_prod_arg[neg_index] / np.abs(arg_l[neg_index]))

        log_cube_prob[pos_index] = (arg[pos_index] ** 2. / (2. * random_var)) - (
        arg[pos_index] * self.lagrange[pos_index] / random_var) + \
                                   (self.lagrange[pos_index] ** 2. / (2. * random_var)) \
                                   - np.log(
            1. / np.abs(arg_l[pos_index]) - prod_arg[pos_index] / np.abs(arg_u[pos_index]))

        neg_log_cube_prob = log_cube_prob.sum()

        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-normal.pdf(arg_u[indicator]) + normal.pdf(arg_l[indicator]),
                                                   cube_prob[indicator])) / self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index]) /
                                    ((prod_arg[pos_index] / np.abs(arg_u[pos_index])) -
                                     (1. / np.abs(arg_l[pos_index])))) / self.randomization_scale

        log_cube_grad[neg_index] = ((-1. + neg_prod_arg[neg_index]) /
                                    ((-neg_prod_arg[neg_index] / np.abs(arg_l[neg_index])) +
                                     (1. / np.abs(arg_u[neg_index])))) / self.randomization_scale

        if mode == 'func':
            return self.scale(neg_log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(neg_log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")

class approximate_conditional_prob(rr.smooth_atom):

    def __init__(self,
                 t, #point at which density is to computed
                 map,
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t
        self.map = map
        self.q = map.p - map.nactive
        self.inactive_conjugate = self.active_conjugate = map.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = self.map.inactive_lagrange

        rr.smooth_atom.__init__(self,
                                (map.nactive,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.map.feasible_point,
                                coef=coef)

        self.coefs[:] = map.feasible_point

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.map.nactive)


    def sel_prob_smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        data = np.squeeze(self.t *  self.map.A)

        offset_active = self.map.offset_active + data[:self.map.nactive]
        offset_inactive = self.map.offset_inactive + data[self.map.nactive:]

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.map.B_active, offset_active))


        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = self.map.randomization_scale)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.map.B_inactive, offset_inactive))

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

    def minimize2(self, step=1, nstep=30, tol=1.e-6):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.sel_prob_smooth_objective(u, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, 'grad')

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

class approximate_conditional_density(rr.smooth_atom):

    def __init__(self, sel_alg,
                       coef=1.,
                       offset=None,
                       quadratic=None,
                       nstep=10):

        self.sel_alg = sel_alg

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

        self.target_observed = self.sel_alg.target_observed
        self.nactive = self.target_observed.shape[0]
        self.target_cov = self.sel_alg.target_cov

    def solve_approx(self):

        #defining the grid on which marginal conditional densities will be evaluated
        self.grid_length = 241

        self.ind_obs = np.zeros(self.nactive, int)
        self.norm = np.zeros(self.nactive)
        self.h_approx = np.zeros((self.nactive, self.grid_length))
        self.grid = np.zeros((self.nactive, self.grid_length))

        for j in range(self.nactive):
            obs = self.target_observed[j]

            self.grid[j,:] = np.linspace(self.target_observed[j]-12., self.target_observed[j]+12.,num=self.grid_length)

            self.norm[j] = self.target_cov[j,j]
            if obs < self.grid[j,0]:
                self.ind_obs[j] = 0
            elif obs > np.max(self.grid[j,:]):
                self.ind_obs[j] = self.grid_length-1
            else:
                self.ind_obs[j] = np.argmin(np.abs(self.grid[j,:]-obs))

            self.h_approx[j, :] = self.approx_conditional_prob(j)

    def approx_conditional_prob(self, j):
        h_hat = []

        self.sel_alg.setup_map(j)

        for i in range(self.grid[j, :].shape[0]):
            approx = approximate_conditional_prob((self.grid[j, :])[i], self.sel_alg)
            val = -(approx.minimize2(step=1, nstep=200)[::-1])[0]

            if val != -float('Inf'):
                h_hat.append(val)
            elif val == -float('Inf') and i == 0:
                h_hat.append(-500.)
            elif val == -float('Inf') and i > 0:
                h_hat.append(h_hat[i - 1])

        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.
        approx_nonnormalized = []
        grad_normalizer = 0.

        for i in range(self.grid_length):
            approx_density = np.exp(-np.true_divide(((self.grid[j,:])[i] - mean) ** 2, 2 * self.norm[j])
                                    + (self.h_approx[j,:])[i])
            normalizer += approx_density
            grad_normalizer += (-mean / self.norm[j] + (self.grid[j, :])[i] / self.norm[j]) * approx_density
            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer)), normalizer, grad_normalizer

    def smooth_objective_MLE(self, param, j, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        approx_normalizer = self.area_normalized_density(j, param)

        f = (param ** 2) / (2 * self.norm[j]) - (self.target_observed[j] * param) / self.norm[j] + \
            log(approx_normalizer[1])

        g = param / self.norm[j] - self.target_observed[j] / self.norm[j] + \
            approx_normalizer[2] / approx_normalizer[1]

        if mode == 'func':
            return self.scale(f)
        elif mode == 'grad':
            return self.scale(g)
        elif mode == 'both':
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def approx_MLE_solver(self, j, step=1, nstep=150, tol=1.e-5):

        current = self.target_observed[j]
        current_value = np.inf

        objective = lambda u: self.smooth_objective_MLE(u, j, 'func')
        grad = lambda u: self.smooth_objective_MLE(u, j, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current) * self.norm[j]

            # make sure proposal is a descent
            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)

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

        value = objective(current)

        return current, value

    def approximate_ci(self, j):

        grid_num = 301
        param_grid = np.linspace(-15.,15., num=grid_num)
        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):
            area_vec = self.area_normalized_density(j, param_grid[k])[0]
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]
        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0

    def approximate_pvalue(self, j, param):

        area_vec = self.area_normalized_density(j, param)[0]
        area = area_vec[self.ind_obs[j]]

        return 2*min(area, 1.-area)
