from __future__ import division, print_function

import numpy as np
from scipy.stats import norm as ndist

from ..distributions.discrete_family import discrete_family
from .approx_reference import grid_inference

class exact_grid_inference(grid_inference):

    def log_reference(self,
                      observed_target,
                      cov_target,
                      linear_coef,
                      grid):

        QS = self.query_spec
        TS = self.target_spec ## we don't use this; it seems that we have already formed the target_specific elements which we input as arguments for this functions

        if np.asarray(observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        ref_hat = []

        cond_precision = np.linalg.inv(QS.cond_cov)
        num_opt = cond_precision.shape[0]
        num_con = QS.linear_part.shape[0]

        for k in range(grid.shape[0]):
            # in the usual D = N + Gamma theta.hat,
            # regress_opt_target is "something" times Gamma,
            # where "something" comes from implied Gaussian
            # cond_mean is "something" times D
            # Gamma is cov_target_score.T.dot(prec_target)

            cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) +
                              QS.cond_mean)

            #direction for decomposing o

            eta = cond_precision.dot(linear_coef).dot(cov_target)

            implied_mean = np.asscalar(eta.T.dot(cond_mean_grid))
            implied_cov = np.asscalar(eta.T.dot(QS.cond_cov).dot(eta))
            implied_prec = 1./implied_cov

            _A = QS.cond_cov.dot(eta) * implied_prec
            R = np.identity(num_opt) - _A.dot(eta.T)

            A = QS.linear_part.dot(_A).reshape((-1,))
            b = -QS.linear_part.dot(R).dot(QS.observed_soln)

            trunc_ = np.true_divide((QS.offset + b), A)

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

        QS = self.query_spec
        TS = self.target_spec

        precs, S, r, T = self.conditional_spec

        self._families = []

        for m in range(self.ntarget):

            observed_target_uni = (TS.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(TS.cov_target)[m]).reshape((1, 1))

            var_target = 1. / (precs[m][0, 0])

            log_ref = self.log_reference(observed_target_uni,
                                         cov_target_uni,
                                         T[m],
                                         self.stat_grid[m])

            logW = (log_ref - 0.5 * (self.stat_grid[m] - TS.observed_target[m]) ** 2 / var_target)
            logW -= logW.max()
            self._families.append(discrete_family(self.stat_grid[m],
                                                  np.exp(logW)))




