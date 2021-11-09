from __future__ import division, print_function

import numpy as np, pandas as pd
from scipy.stats import norm as ndist

from ..distributions.discrete_family import discrete_family

class exact_grid_inference(object):

    def __init__(self,
                 query,
                 target_spec,
                 solve_args={'tol': 1.e-12}):

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
        cov_target : ndarray
            Estimated covaraince of target.
        cov_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        self.solve_args = solve_args

        (observed_target,
         cov_target,
         regress_target_score) = target_spec[:3]

        self.observed_target = observed_target
        self.cov_target = cov_target
        self.prec_target = np.linalg.inv(cov_target)
        self.regress_target_score = regress_target_score

        self.cond_mean = query.cond_mean
        self.cond_cov = query.cond_cov
        self.cond_precision = np.linalg.inv(self.cond_cov)
        self.opt_linear = query.opt_linear

        self.linear_part = query.affine_con.linear_part
        self.offset = query.affine_con.offset

        self.M1 = query.M1
        self.M2 = query.M2
        self.M3 = query.M3
        self.observed_soln = query.observed_opt_state

        self.observed_score = query.observed_score_state + query.observed_subgrad

        result, inverse_info, log_ref = query.selective_MLE(target_spec)

        self.ntarget = ntarget = cov_target.shape[0]
        _scale = 4 * np.sqrt(np.diag(inverse_info))

        ngrid = 1000
        self.stat_grid = np.zeros((ntarget, ngrid))
        for j in range(ntarget):
            self.stat_grid[j, :] = np.linspace(observed_target[j] - 1.5 * _scale[j],
                                               observed_target[j] + 1.5 * _scale[j],
                                               num=ngrid)

        self.inverse_info = inverse_info

    def summary(self,
                alternatives=None,
                parameter=None,
                level=0.90):
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
                      cov_target,
                      linear_coef,
                      grid):

        if np.asarray(observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        ref_hat = []

        for k in range(grid.shape[0]):
            # in the usual D = N + Gamma theta.hat,
            # regress_opt_target is "something" times Gamma,
            # where "something" comes from implied Gaussian
            # cond_mean is "something" times D
            # Gamma is cov_target_score.T.dot(prec_target)

            num_opt = self.cond_precision.shape[0]
            num_con = self.linear_part.shape[0]

            cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) +
                              self.cond_mean)

            #direction for decomposing o

            eta = self.cond_precision.dot(linear_coef).dot(cov_target)

            implied_mean = np.asscalar(eta.T.dot(cond_mean_grid))
            implied_cov = np.asscalar(eta.T.dot(self.cond_cov).dot(eta))
            implied_prec = 1./implied_cov

            _A = self.cond_cov.dot(eta) * implied_prec
            R = np.identity(num_opt) - _A.dot(eta.T)

            A = self.linear_part.dot(_A).reshape((-1,))
            b = -self.linear_part.dot(R).dot(self.observed_soln)

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

            observed_target_uni = (self.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(self.cov_target)[m]).reshape((1, 1))

            var_target = 1. / ((self.precs[m])[0, 0])

            log_ref = self.log_reference(observed_target_uni,
                                         cov_target_uni,
                                         self.T[m],
                                         self.stat_grid[m])

            logW = (log_ref - 0.5 * (self.stat_grid[m] - self.observed_target[m]) ** 2 / var_target)
            logW -= logW.max()
            self._families.append(discrete_family(self.stat_grid[m],
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
        T = {}

        p = self.regress_target_score.shape[1]

        for m in range(self.ntarget):
            observed_target_uni = (self.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(self.cov_target)[m]).reshape((1, 1))
            prec_target = 1. / cov_target_uni
            regress_target_score_uni = self.regress_target_score[m, :].reshape((1, p))

            T1 = regress_target_score_uni.T.dot(prec_target)
            T2 = T1.T.dot(self.M2.dot(T1))
            T3 = T1.T.dot(self.M3.dot(T1))
            T4 = self.M1.dot(self.opt_linear).dot(self.cond_cov).dot(self.opt_linear.T.dot(self.M1.T.dot(T1)))
            T5 = T1.T.dot(self.M1.dot(self.opt_linear))

            _T = self.cond_cov.dot(T5.T)

            prec_target_nosel = prec_target + T2 - T3

            _P = -(T1.T.dot(self.M1.dot(self.observed_score)) + T2.dot(observed_target_uni))

            bias_target = cov_target_uni.dot(T1.T.dot(-T4.dot(observed_target_uni) + self.M1.dot(self.opt_linear.dot(self.cond_mean))) - _P)

            _r = np.linalg.inv(prec_target_nosel).dot(prec_target.dot(bias_target))
            _S = np.linalg.inv(prec_target_nosel).dot(prec_target)

            S[m] = _S
            r[m] = _r
            precs[m] = prec_target_nosel
            T[m] = _T

        self.precs = precs
        self.S = S
        self.r = r
        self.T = T




