from __future__ import division, print_function

import numpy as np, sys
from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C


class approximate_grid_inference():

    def __init__(self,
                 query,
                 observed_target,
                 cov_target,
                 cov_target_score,
                 grid,
                 dispersion=1,
                 level=0.9,
                 solve_args={'tol':1.e-12}):

        self.solve_args = solve_args

        self.linear_part = query.sampler.affine_con.linear_part
        self.offset = query.sampler.affine_con.offset

        self.logdens_linear = query.sampler.logdens_transform[0]
        self.cond_mean = query.cond_mean
        self.prec_opt = np.linalg.inv(query.cond_cov)
        self.cond_cov = query.cond_cov

        self.observed_target = observed_target
        self.cov_target_score = cov_target_score
        self.cov_target = cov_target

        self.init_soln = query.observed_opt_state
        self.grid = grid

        self.ntarget = cov_target.shape[0]
        self.level = level

    def approx_log_reference(self,
                             observed_target,
                             cov_target,
                             cov_target_score):

        if np.asarray(observed_target).shape in [(), (0,)]:
           raise ValueError('no target specified')

        observed_target = np.atleast_1d(observed_target)
        prec_target = np.linalg.inv(cov_target)
        target_lin = - self.logdens_linear.dot(cov_target_score.T.dot(prec_target))

        ref_hat = []
        solver = solve_barrier_affine_C
        for k in range(self.grid.shape[0]):
            cond_mean_grid = target_lin.dot(np.asarray([self.grid[k]])) + (
                    self.cond_mean - target_lin.dot(observed_target))
            conjugate_arg = self.prec_opt.dot(cond_mean_grid)

            val, _, _ = solver(conjugate_arg,
                               self.prec_opt,
                               self.init_soln,
                               self.linear_part,
                               self.offset,
                               **self.solve_args)

            ref_hat.append(-val - (conjugate_arg.T.dot(self.cond_cov).dot(conjugate_arg) / 2.))

        return np.asarray(ref_hat)


    def approx_density(self,
                       mean_parameter,
                       cov_target,
                       approx_log_ref):

        _approx_density = []
        for k in range(self.grid.shape[0]):
            _approx_density.append(np.exp(-np.true_divide((self.grid[k] - mean_parameter) ** 2, 2 * cov_target) + approx_log_ref[k]))

        _approx_density_ = np.asarray(_approx_density) / (np.asarray(_approx_density).sum())
        return np.cumsum(_approx_density_)

    def approx_ci(self,
                  param_grid,
                  cov_target,
                  approx_log_ref,
                  indx_obsv):

        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):
            area_vec = self.approx_density(param_grid[k],
                                           cov_target,
                                           approx_log_ref)

            area[k] = area_vec[indx_obsv]

        alpha = 1 - self.level
        region = param_grid[(area >= alpha / 2.) & (area <= (1 - alpha / 2.))]

        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0., 0.

    def approx_pivot(self,
                     mean_parameter):

        pivot = []

        for m in range(self.ntarget):
            p = self.cov_target_score.shape[1]
            observed_target_uni = (self.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(self.cov_target)[m]).reshape((1, 1))
            cov_target_score_uni = self.cov_target_score[m, :].reshape((1, p))
            grid_indx_obs = np.argmin(np.abs(self.grid - observed_target_uni))

            approx_log_ref = self.approx_log_reference(observed_target_uni,
                                                       cov_target_uni,
                                                       cov_target_score_uni)

            area_cum = self.approx_density(mean_parameter[m],
                                           cov_target_uni,
                                           approx_log_ref)

            pivot.append(2 * np.minimum(area_cum[grid_indx_obs], 1. - area_cum[grid_indx_obs]))

            sys.stderr.write("variable completed " + str(m + 1) + "\n")

        return pivot

    def approx_intervals(self,
                         param_grid):

        intervals_lci =[]
        intervals_uci =[]

        for m in range(self.ntarget):
            p = self.cov_target_score.shape[1]
            observed_target_uni = (self.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(self.cov_target)[m]).reshape((1, 1))
            cov_target_score_uni = self.cov_target_score[m, :].reshape((1, p))
            grid_indx_obs = np.argmin(np.abs(self.grid - observed_target_uni))

            approx_log_ref = self.approx_log_reference(observed_target_uni,
                                                       cov_target_uni,
                                                       cov_target_score_uni)

            approx_lci, approx_uci = self.approx_ci(param_grid[m,:],
                                                    cov_target_uni,
                                                    approx_log_ref,
                                                    grid_indx_obs)

            intervals_lci.append(approx_lci)
            intervals_uci.append(approx_uci)

            sys.stderr.write("variable completed " + str(m + 1) + "\n")

        return np.asarray(intervals_lci), np.asarray(intervals_uci)




