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

        self.query_spec = query_spec
        self.target_spec = target_spec
        self.solve_args = solve_args
        
    def solve_estimating_eqn(self,
                             alternatives=None,
                             useC=False,
                             level=0.90):

        prec_target_nosel, bias_target, U3, U5 = _setup_estimating_eqn(self.query_spec,
                                                                       self.target_spec)

        Q = self.query_spec
        TS = self.target_spec
        
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

        final_estimator = TS.cov_target.dot(prec_target_nosel).dot(TS.observed_target) \
                          + TS.regress_target_score.dot(Q.M1.dot(Q.opt_linear)).dot(Q.cond_mean - soln) \
                          - bias_target

        observed_info_natural = prec_target_nosel + U3 - U5.dot(hess.dot(U5.T))

        unbiased_estimator = TS.cov_target.dot(prec_target_nosel).dot(TS.observed_target) - bias_target

        observed_info_mean = TS.cov_target.dot(observed_info_natural.dot(TS.cov_target))

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

def _setup_estimating_eqn(query_spec,
                          target_spec):

        Q = query_spec
        TS = target_spec

        prec_target = np.linalg.inv(TS.cov_target)
        U1 = TS.regress_target_score.T.dot(prec_target)
        U2 = U1.T.dot(Q.M2.dot(U1))
        U3 = U1.T.dot(Q.M3.dot(U1))
        U4 = Q.M1.dot(Q.opt_linear).dot(Q.cond_cov).dot(Q.opt_linear.T.dot(Q.M1.T.dot(U1)))
        U5 = U1.T.dot(Q.M1.dot(Q.opt_linear))

        prec_target_nosel = prec_target + U2 - U3

        _P = -(U1.T.dot(Q.M1.dot(Q.observed_score)) + U2.dot(TS.observed_target))

        bias_target = TS.cov_target.dot(U1.T.dot(-U4.dot(TS.observed_target)
                                                   + Q.M1.dot(Q.opt_linear.dot(Q.cond_mean))) - _P)

        return prec_target_nosel, bias_target, U3, U5









