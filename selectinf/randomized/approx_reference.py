from __future__ import division, print_function

import numpy as np, pandas as pd
from scipy.interpolate import interp1d

from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from ..distributions.discrete_family import discrete_family

class approximate_grid_inference(object):

    def __init__(self,
                 query,
                 observed_target,
                 cov_target,
                 cov_target_score,
                 grid=None,
                 solve_args={'tol':1.e-12}):

        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        observed_target : ndarray
            Observed estimate of target.

        target_cov : ndarray
            Estimated covaraince of target.

        target_score_cov : ndarray
            Estimated covariance of target and score of randomized query.

        grid : ndarray
            Grid on which to evaluate the approximate
            probability of selection.

        mle : ndarray
            Selective MLE as initial guess.

        inverse_info : ndarray
            Selective inverse information to guide grid search.

        """

        self.solve_args = solve_args

        result, inverse_info = query.selective_MLE(observed_target,
                                                   cov_target,
                                                   cov_target_score,
                                                   solve_args=solve_args)[:2]
        mle = result['MLE']
        
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

        self.ntarget = ntarget = cov_target.shape[0]
        _scale = 4 * np.sqrt(np.diag(inverse_info))
        ngrid = 40

        scale_ = 4 * np.max(np.sqrt(np.diag(inverse_info)))

        self.stat_grid = np.zeros((ntarget, ngrid))
        for j in range(ntarget):
            self.stat_grid[j,:] = np.linspace(observed_target[j] - 1.5*_scale[j],
                                              observed_target[j] + 1.5*_scale[j],
                                              num=ngrid)


    def _approx_log_reference(self,
                             observed_target,
                             cov_target,
                             cov_target_score,
                             grid):

        """
        Approximate the log of the reference density on a grid.

        """
        if np.asarray(observed_target).shape in [(), (0,)]:
           raise ValueError('no target specified')

        prec_target = np.linalg.inv(cov_target)
        target_lin = - self.logdens_linear.dot(cov_target_score.T.dot(prec_target))

        ref_hat = []
        solver = solve_barrier_affine_C
        for k in range(grid.shape[0]):
            # in the usual D = N + Gamma theta.hat,
            # target_lin is "something" times Gamma,
            # where "something" comes from implied Gaussian
            # cond_mean is "something" times D
            # Gamma is cov_target_score.T.dot(prec_target)
            
            cond_mean_grid = (target_lin.dot(np.atleast_1d(grid[k] - observed_target)) + 
                              self.cond_mean)
            conjugate_arg = self.prec_opt.dot(cond_mean_grid)

            val, _, _ = solver(conjugate_arg,
                               self.prec_opt,
                               self.init_soln,
                               self.linear_part,
                               self.offset,
                               **self.solve_args)

            ref_hat.append(-val - (conjugate_arg.T.dot(self.cond_cov).dot(conjugate_arg) / 2.))

        return np.asarray(ref_hat)

    def approx_CDF(self,
                   mean_parameter,
                   cov_target,
                   approx_log_ref,
                   grid):

        _approx_density = []
        for k in range(grid.shape[0]):
            # approx_log_ref[k] = P(selection | D = N + Gamma * grid[k])
            _approx_density.append(np.exp(-np.true_divide((grid[k] - mean_parameter)**2,
                                                          2 * cov_target) + approx_log_ref[k]))

        _approx_density_ = np.asarray(_approx_density) / (np.asarray(_approx_density).sum())
        return np.cumsum(_approx_density_)

    def approx_ci(self,
                  param_grid,
                  stat_grid,
                  cov_target,
                  approx_log_ref,
                  indx_obsv,
                  level):

        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):
            area_vec = self.approx_CDF(param_grid[k],
                                       cov_target,
                                       approx_log_ref,
                                       stat_grid)

            area[k] = area_vec[indx_obsv]

        alpha = 1 - level
        region = param_grid[(area >= alpha / 2.) & (area <= (1 - alpha / 2.))]

        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0., 0.

    def _construct_families(self):

        self._families = []
        for m in range(self.ntarget):
            p = self.cov_target_score.shape[1]
            observed_target_uni = (self.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(self.cov_target)[m]).reshape((1, 1))
            var_target = cov_target_uni[0, 0]
            cov_target_score_uni = self.cov_target_score[m, :].reshape((1, p))

            approx_log_ref = self._approx_log_reference(observed_target_uni,
                                                        cov_target_uni,
                                                        cov_target_score_uni,
                                                        self.stat_grid[m])

            approx_fn = interp1d(self.stat_grid[m],
                                 approx_log_ref,
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')

            grid = np.linspace(self.stat_grid[m].min(), self.stat_grid[m].max(), 1000)
            logW = (approx_fn(grid) -
                    0.5 * (grid - self.observed_target[m])**2 / var_target)
            logW -= logW.max()

            self._families.append(discrete_family(grid,
                                                  np.exp(logW)))
            
            logG = - 0.5 * grid**2 / var_target
            logG -= logG.max()
            import matplotlib.pyplot as plt

            # plt.plot(self.stat_grid[m][10:30], approx_log_ref[10:30])
            # plt.plot(self.stat_grid[m][:10], approx_log_ref[:10], 'r', linewidth=4)
            # plt.plot(self.stat_grid[m][30:], approx_log_ref[30:], 'r', linewidth=4)
            # plt.plot(self.stat_grid[m]*1.5, fapprox(self.stat_grid[m]*1.5), 'k--')
            # plt.show()

            # plt.plot(grid, logW)
            # plt.plot(grid, logG)

            # stop

    def approx_pivots(self,
                      mean_parameter,
                      alternatives=None):

        if not hasattr(self, "_families"):
            self._construct_families()
            
        if alternatives is None:
            alternatives = ['twosided'] * self.ntarget

        pivot = []

        for m in range(self.ntarget):
            family = self._families[m]
            observed_target = self.observed_target[m]
            var_target = self.cov_target[m, m]
            _cdf = family.cdf((mean_parameter[m] - observed_target) / var_target,
                              x=observed_target)
            if alternatives[m] == 'twosided':
                pivot.append(2 * min(_cdf, 1 - _cdf))
            elif alternatives[m] == 'greater':
                pivot.append(1 - _cdf)
            elif alternatives[m] == 'less':
                pivot.append(_cdf)
            else:
                raise ValueError('alternative should be in ["twosided", "less", "greater"]')
        return pivot

    def approx_intervals(self,
                         level=0.9):

        if not hasattr(self, "_families"):
            self._construct_families()
            
        lower, upper = [], []

        for m in range(self.ntarget):
            family = self._families[m]
            observed_target = self.observed_target[m]
            l, u = family.equal_tailed_interval(observed_target,
                                                        alpha=1-level)
            var_target = self.cov_target[m, m]
            lower.append(l *  var_target + observed_target)
            upper.append(u * var_target + observed_target)

        return np.asarray(lower), np.asarray(upper)

    def summary(self,
                alternatives=None,
                parameter=None,
                level=0.9):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']

        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.

        level : float
            Confidence level.

        """

        if parameter is not None:
            pivots = self.approx_pivots(parameter,
                                        alternatives=alternatives)
        else:
            pivots = None

        pvalues = self.approx_pivots(np.zeros_like(self.observed_target),
                                     alternatives=alternatives)
        lower, upper = self.approx_intervals(level=level)

        result = pd.DataFrame({'target':self.observed_target,
                               'pvalue':pvalues,
                               'lower_confidence':lower,
                               'upper_confidence':upper})

        if not np.all(parameter == 0):
            result.insert(4, 'pivot', pivots)
            result.insert(5, 'parameter', parameter)

        return result

def _log_concave_approx(xval, yval):
    """
    Approximate a log-concave function
    to full line based on sample.

    Assumes `xval` is sorted
    """

    nu, nl = 10, 10
    n = xval.shape[0]
    D = np.vstack([np.ones(n), xval, xval**2]).T

    Du = D[-nu:]
    Qu = np.linalg(Du).dot(yval[-nu:])

    Dl = D[:nl]
    Ql = np.linalg(Dl).dot(yval[:nl])
