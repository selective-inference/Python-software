from __future__ import division, print_function

import numpy as np, pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm as ndist

from ..distributions.discrete_family import discrete_family

class exact_grid_inference(object):

    def __init__(self,
                 query,
                 observed_target,
                 target_cov,
                 target_score_cov,
                 solve_args={'tol': 1.e-12},
                 useIP=False):

        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        query : `gaussian_query`
            A Gaussian query which has information
            to describe implied Gaussian.
        observed_target : ndarray
            Observed estimate of target.
        target_cov : ndarray
            Estimated covaraince of target.
        target_score_cov : ndarray
            Estimated covariance of target and score of randomized query.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        result, inverse_info = query.selective_MLE(observed_target,
                                                   target_cov,
                                                   target_score_cov,
                                                   solve_args=solve_args)[:2]

        self.linear_part = query.sampler.affine_con.linear_part
        self.offset = query.sampler.affine_con.offset

        self.logdens_linear = query.sampler.logdens_transform[0]
        self.cond_mean = query.cond_mean
        self.prec_opt = np.linalg.inv(query.cond_cov)
        self.cond_cov = query.cond_cov

        self.observed_target = observed_target
        self.target_score_cov = target_score_cov
        self.target_cov = target_cov

        self.init_soln = query.observed_opt_state

        self.randomizer_prec = query.sampler.randomizer_prec
        self.score_offset = query.observed_score_state + query.sampler.logdens_transform[1]

        self.ntarget = ntarget = target_cov.shape[0]
        _scale = 4 * np.sqrt(np.diag(inverse_info))

        if useIP == False:
            ngrid = 1000
            self.stat_grid = np.zeros((ntarget, ngrid))
            for j in range(ntarget):
                self.stat_grid[j, :] = np.linspace(observed_target[j] - 1.5 * _scale[j],
                                                   observed_target[j] + 1.5 * _scale[j],
                                                   num=ngrid)
        else:
            ngrid = 60
            self.stat_grid = np.zeros((ntarget, ngrid))
            for j in range(ntarget):
                self.stat_grid[j, :] = np.linspace(observed_target[j] - 1.5 * _scale[j],
                                                   observed_target[j] + 1.5 * _scale[j],
                                                   num=ngrid)

        self.opt_linear = query.opt_linear
        self.useIP = useIP

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
            pivots = self._pivots(parameter,
                                        alternatives=alternatives)
        else:
            pivots = None

        pvalues = self._pivots(np.zeros_like(self.observed_target),
                                      alternatives=alternatives)
        lower, upper = self._intervals(level=level)

        result = pd.DataFrame({'target': self.observed_target,
                               'pvalue': pvalues,
                               'lower_confidence': lower,
                               'upper_confidence': upper})

        if not np.all(parameter == 0):
            result.insert(4, 'pivot', pivots)
            result.insert(5, 'parameter', parameter)

        return result

    def log_reference(self,
                      observed_target,
                      target_cov,
                      target_score_cov,
                      grid):

        if np.asarray(observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        prec_target = np.linalg.inv(target_cov)
        target_lin = - self.logdens_linear.dot(target_score_cov.T.dot(prec_target))

        ref_hat = []

        for k in range(grid.shape[0]):
            # in the usual D = N + Gamma theta.hat,
            # target_lin is "something" times Gamma,
            # where "something" comes from implied Gaussian
            # cond_mean is "something" times D
            # Gamma is target_score_cov.T.dot(prec_target)

            num_opt = self.prec_opt.shape[0]
            num_con = self.linear_part.shape[0]

            cond_mean_grid = (target_lin.dot(np.atleast_1d(grid[k] - observed_target)) +
                              self.cond_mean)

            #direction for decomposing o

            eta = -self.prec_opt.dot(self.logdens_linear.dot(target_score_cov.T))

            implied_mean = np.asscalar(eta.T.dot(cond_mean_grid))
            implied_cov = np.asscalar(eta.T.dot(self.cond_cov).dot(eta))
            implied_prec = 1./implied_cov

            _A = self.cond_cov.dot(eta) * implied_prec
            R = np.identity(num_opt) - _A.dot(eta.T)

            A = self.linear_part.dot(_A).reshape((-1,))
            b = -self.linear_part.dot(R).dot(self.init_soln)

            trunc_ = np.true_divide((self.offset + b), A)

            neg_indx = np.asarray([j for j in range(num_con) if A[j] < 0.])
            pos_indx = np.asarray([j for j in range(num_con) if A[j] > 0.])

            if pos_indx.shape[0]>0 and neg_indx.shape[0]>0:

                trunc_lower = np.max(trunc_[neg_indx])
                trunc_upper = np.min(trunc_[pos_indx])

                lower_limit = (trunc_lower - implied_mean) * np.sqrt(implied_prec)
                upper_limit = (trunc_upper - implied_mean) * np.sqrt(implied_prec)

                ref_hat.append(np.log(ndist.cdf(upper_limit) - ndist.cdf(lower_limit)))

            elif pos_indx.shape[0] == num_con:

                trunc_upper = np.min(trunc_[pos_indx])

                upper_limit = (trunc_upper - implied_mean) * np.sqrt(implied_prec)

                ref_hat.append(np.log(ndist.cdf(upper_limit)))

            else:

                trunc_lower = np.max(trunc_[neg_indx])

                lower_limit = (trunc_lower - implied_mean) * np.sqrt(implied_prec)

                ref_hat.append(np.log(1. - ndist.cdf(lower_limit)))

        return np.asarray(ref_hat)

    def _construct_families(self):

        self._construct_density()

        self._families = []

        for m in range(self.ntarget):
            p = self.target_score_cov.shape[1]
            observed_target_uni = (self.observed_target[m]).reshape((1,))

            target_cov_uni = (np.diag(self.target_cov)[m]).reshape((1, 1))
            target_score_cov_uni = self.target_score_cov[m, :].reshape((1, p))

            var_target = 1. / ((self.precs[m])[0, 0])

            log_ref = self.log_reference(observed_target_uni,
                                         target_cov_uni,
                                         target_score_cov_uni,
                                         self.stat_grid[m])
            if self.useIP == False:
                logW = (log_ref - 0.5 * (self.stat_grid[m] - self.observed_target[m]) ** 2 / var_target)
                logW -= logW.max()
                self._families.append(discrete_family(self.stat_grid[m],
                                                      np.exp(logW)))
            else:
                approx_fn = interp1d(self.stat_grid[m],
                                     log_ref,
                                     kind='quadratic',
                                     bounds_error=False,
                                     fill_value='extrapolate')

                grid = np.linspace(self.stat_grid[m].min(), self.stat_grid[m].max(), 1000)
                logW = (approx_fn(grid) -
                        0.5 * (grid - self.observed_target[m]) ** 2 / var_target)

                logW -= logW.max()
                self._families.append(discrete_family(grid,
                                                      np.exp(logW)))

    def _pivots(self,
                mean_parameter,
                alternatives=None):

        if not hasattr(self, "_families"):
            self._construct_families()

        if alternatives is None:
            alternatives = ['twosided'] * self.ntarget

        pivot = []

        for m in range(self.ntarget):

            family = self._families[m]
            var_target = 1. / ((self.precs[m])[0, 0])

            mean = self.S[m].dot(mean_parameter[m].reshape((1,))) + self.r[m]

            _cdf = family.cdf((mean[0] - self.observed_target[m]) / var_target, x=self.observed_target[m])

            if alternatives[m] == 'twosided':
                pivot.append(2 * min(_cdf, 1 - _cdf))
            elif alternatives[m] == 'greater':
                pivot.append(1 - _cdf)
            elif alternatives[m] == 'less':
                pivot.append(_cdf)
            else:
                raise ValueError('alternative should be in ["twosided", "less", "greater"]')
        return pivot

    def _intervals(self,
                   level=0.9):

        if not hasattr(self, "_families"):
            self._construct_families()

        lower, upper = [], []

        for m in range(self.ntarget):
            # construction of intervals from families follows `selectinf.learning.core`
            family = self._families[m]
            observed_target = self.observed_target[m]

            l, u = family.equal_tailed_interval(observed_target,
                                                alpha=1 - level)

            var_target = 1. / ((self.precs[m])[0, 0])

            lower.append(l * var_target + observed_target)
            upper.append(u * var_target + observed_target)

        return np.asarray(lower), np.asarray(upper)

    ### Private method
    def _construct_density(self):

        precs = {}
        S = {}
        r = {}

        p = self.target_score_cov.shape[1]

        for m in range(self.ntarget):
            observed_target_uni = (self.observed_target[m]).reshape((1,))
            target_cov_uni = (np.diag(self.target_cov)[m]).reshape((1, 1))
            prec_target = 1. / target_cov_uni
            target_score_cov_uni = self.target_score_cov[m, :].reshape((1, p))

            target_linear = target_score_cov_uni.T.dot(prec_target)
            target_offset = (self.score_offset - target_linear.dot(observed_target_uni)).reshape(
                (target_linear.shape[0],))

            target_lin = -self.logdens_linear.dot(target_linear)
            target_off = (self.cond_mean - target_lin.dot(observed_target_uni)).reshape((target_lin.shape[0],))

            _prec = prec_target + (target_linear.T.dot(target_linear) * self.randomizer_prec) - target_lin.T.dot(
                self.prec_opt).dot(target_lin)

            _P = target_linear.T.dot(target_offset) * self.randomizer_prec
            _r = (1. / _prec).dot(target_lin.T.dot(self.prec_opt).dot(target_off) - _P)
            _S = np.linalg.inv(_prec).dot(prec_target)

            S[m] = _S
            r[m] = _r
            precs[m] = _prec

        self.precs = precs
        self.S = S
        self.r = r




