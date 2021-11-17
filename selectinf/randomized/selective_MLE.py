from __future__ import division, print_function

import numpy as np, pandas as pd
from scipy.stats import norm as ndist
from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from ..algorithms.barrier_affine import solve_barrier_affine_py

class mle_inference(object):

    def __init__(self,
                 query_spec,
                 target_spec,
                 solve_args={'tol': 1.e-12}):

        self.solve_args = solve_args

        (observed_target,
         cov_target,
         regress_target_score) = target_spec[:3]

        self.observed_target = observed_target
        self.cov_target = cov_target
        self.prec_target = np.linalg.inv(cov_target)
        self.regress_target_score = regress_target_score

        self.query_spec = query_spec

        self._setup_estimating_eqn()

    def solve_estimating_eqn(self,
                             alternatives=None,
                             useC=False,
                             level=0.90):

        Q = self.query_spec
        cond_precision = np.linalg.inv(Q.cond_cov)
        conjugate_arg = cond_precision.dot(Q.cond_mean)

        if useC:
            solver = solve_barrier_affine_C
        else:
            solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_arg,
                                 cond_precision,
                                 Q.observed_soln,
                                 Q.linear_part,
                                 Q.offset,
                                 **self.solve_args)

        final_estimator = self.cov_target.dot(self.prec_target_nosel).dot(self.observed_target) \
                          + self.regress_target_score.dot(Q.M1.dot(Q.opt_linear)).dot(Q.cond_mean - soln) \
                          - self.bias_target

        observed_info_natural = self.prec_target_nosel + self.T3 - self.T5.dot(hess.dot(self.T5.T))

        unbiased_estimator = self.cov_target.dot(self.prec_target_nosel).dot(self.observed_target) - self.bias_target

        observed_info_mean = self.cov_target.dot(observed_info_natural.dot(self.cov_target))

        Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))

        cdf_vals = ndist.cdf(Z_scores)
        pvalues = []

        if alternatives is None:
            alternatives = ['twosided'] * len(cdf_vals)

        for m, _cdf in enumerate(cdf_vals):
            if alternatives[m] == 'twosided':
                pvalues.append(2 * min(_cdf, 1 - _cdf))
            elif alternatives[m] == 'greater':
                pvalues.append(1 - _cdf)
            elif alternatives[m] == 'less':
                pvalues.append(_cdf)
            else:
                raise ValueError('alternative should be in ["twosided", "less", "greater"]')

        alpha = 1. - level

        quantile = ndist.ppf(1 - alpha / 2.)

        intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                               final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

        log_ref = val + conjugate_arg.T.dot(Q.cond_cov).dot(conjugate_arg) / 2.

        result = pd.DataFrame({'MLE': final_estimator,
                               'SE': np.sqrt(np.diag(observed_info_mean)),
                               'Zvalue': Z_scores,
                               'pvalue': pvalues,
                               'alternative': alternatives,
                               'lower_confidence': intervals[:, 0],
                               'upper_confidence': intervals[:, 1],
                               'unbiased': unbiased_estimator})

        return result, observed_info_mean, log_ref

    def _setup_estimating_eqn(self):

        Q = self.query_spec
        T1 = self.regress_target_score.T.dot(self.prec_target)
        T2 = T1.T.dot(Q.M2.dot(T1))
        T3 = T1.T.dot(Q.M3.dot(T1))
        T4 = Q.M1.dot(Q.opt_linear).dot(Q.cond_cov).dot(Q.opt_linear.T.dot(Q.M1.T.dot(T1)))
        T5 = T1.T.dot(Q.M1.dot(Q.opt_linear))

        self.prec_target_nosel = self.prec_target + T2 - T3

        _P = -(T1.T.dot(Q.M1.dot(Q.observed_score)) + T2.dot(self.observed_target))

        self.bias_target = self.cov_target.dot(T1.T.dot(-T4.dot(self.observed_target)
                                                   + Q.M1.dot(Q.opt_linear.dot(Q.cond_mean))) - _P)

        self.T3 = T3
        self.T5 = T5









