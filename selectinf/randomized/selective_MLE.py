from __future__ import division, print_function

import numpy as np, pandas as pd
from scipy.stats import norm as ndist
from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from ..algorithms.barrier_affine import solve_barrier_affine_py

class mle_inference(object):

    def __init__(self,
                 query,
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

        self.cond_mean = query.cond_mean
        self.cond_cov = query.cond_cov
        self.prec_opt = np.linalg.inv(self.cond_cov)
        self.opt_linear = query.opt_linear

        self.linear_part = query.sampler.affine_con.linear_part
        self.offset = query.sampler.affine_con.offset

        self.M1 = query.M1
        self.M2 = query.M2
        self.M3 = query.M3
        self.observed_soln = query.observed_opt_state

        self.observed_score = query.observed_score_state + query.observed_subgrad

        self._setup_estimating_eqn()

    def mle_inference(self, useC= False, level=0.90):

        conjugate_arg = self.prec_opt.dot(self.cond_mean)
        if useC:
            solver = solve_barrier_affine_C
        else:
            solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_arg,
                                 self.prec_opt,
                                 self.observed_soln,
                                 self.linear_part,
                                 self.offset,
                                 **self.solve_args)

        final_estimator = self.cov_target.dot(self.prec_target_nosel).dot(self.observed_target) \
                          + self.regress_target_score.dot(self.M1.dot(self.opt_linear)).dot(self.cond_mean - soln) \
                          - self.bias_target

        observed_info_natural = self.prec_target_nosel + self.T3 - self.T5.dot(self.hess.dot(self.T5.T))

        unbiased_estimator = self.cov_target.dot(self.prec_target_nosel).dot(self.observed_target) - self.bias_target

        observed_info_mean = self.cov_target.dot(observed_info_natural.dot(self.cov_target))

        Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))

        pvalues = ndist.cdf(Z_scores)

        pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

        alpha = 1. - level

        quantile = ndist.ppf(1 - alpha / 2.)

        intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                               final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

        log_ref = val + conjugate_arg.T.dot(self.cond_cov).dot(conjugate_arg) / 2.

        result = pd.DataFrame({'MLE': final_estimator,
                               'SE': np.sqrt(np.diag(observed_info_mean)),
                               'Zvalue': Z_scores,
                               'pvalue': pvalues,
                               'lower_confidence': intervals[:, 0],
                               'upper_confidence': intervals[:, 1],
                               'unbiased': unbiased_estimator})

        return result, observed_info_mean, log_ref

    def _setup_estimating_eqn(self):

        T1 = self.regress_target_score.T.dot(self.prec_target)
        T2 = T1.T.dot(self.M2.dot(T1))
        T3 = T1.T.dot(self.M3.dot(T1))
        T4 = self.M1.dot(self.opt_linear).dot(self.cond_cov).dot(self.opt_linear.T.dot(self.M1.T.dot(T1)))
        T5 = T1.T.dot(self.M1.dot(self.opt_linear))

        self.prec_target_nosel = self.prec_target + T2 - T3

        _P = -(T1.T.dot(self.M1.dot(self.observed_score)) + T2.dot(self.observed_target))

        self.bias_target = self.cov_target.dot(T1.T.dot(-T4.dot(self.observed_target)
                                                   + self.M1.dot(self.opt_linear.dot(self.cond_mean))) - _P)

        self.T3 = T3
        self.T5 = T5









