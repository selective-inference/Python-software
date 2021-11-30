
from __future__ import division, print_function

import numpy as np, pandas as pd
from scipy.interpolate import interp1d

from ..distributions.discrete_family import discrete_family
from ..algorithms.barrier_affine import solve_barrier_affine_py
from .base import grid_inference

class approximate_grid_inference(grid_inference):

    def __init__(self,
                 query_spec,
                 target_spec,
                 solve_args={'tol': 1.e-12},
                 ngrid=1000,
                 ncoarse=40):

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

        grid_inference.__init__(self,
                                query_spec,
                                target_spec,
                                solve_args=solve_args)

        self.ncoarse = ncoarse

    def _approx_log_reference(self,
                              observed_target,
                              cov_target,
                              linear_coef,
                              grid):

        """
        Approximate the log of the reference density on a grid.
        """

        TS = self.target_spec
        QS = self.query_spec
        cond_precision = np.linalg.inv(QS.cond_cov)
        
        if np.asarray(observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        ref_hat = []
        solver = solve_barrier_affine_py

        for k in range(grid.shape[0]):
            # in the usual D = N + Gamma theta.hat,
            # regress_opt_target is "something" times Gamma,
            # where "something" comes from implied Gaussian
            # cond_mean is "something" times D
            # Gamma is cov_target_score.T.dot(prec_target)

            cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) + QS.cond_mean)
            conjugate_arg = cond_precision.dot(cond_mean_grid)

            val, _, _ = solver(conjugate_arg,
                               cond_precision,
                               QS.observed_soln,
                               QS.linear_part,
                               QS.offset,
                               **self.solve_args)

            ref_hat.append(-val - (conjugate_arg.T.dot(QS.cond_cov).dot(conjugate_arg) / 2.))

        return np.asarray(ref_hat)

    def _construct_families(self):

        TS = self.target_spec
        QS = self.query_spec

        precs, S, r, T = self.conditional_spec

        self._families = []

        if self.ncoarse is not None:
            coarse_grid = np.zeros((self.stat_grid.shape[0], self.ncoarse))
            for j in range(coarse_grid.shape[0]):
                coarse_grid[j,:] = np.linspace(self.stat_grid[j].min(),
                                               self.stat_grid[j].max(),
                                               self.ncoarse)
            eval_grid = coarse_grid
        else:
            eval_grid = self.stat_grid
            
        _log_ref = np.zeros((self.ntarget, self.stat_grid[0].shape[0]))

        for m in range(self.ntarget):

            observed_target_uni = (TS.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(TS.cov_target)[m]).reshape((1, 1))

            var_target = 1. / (precs[m][0, 0])

            approx_log_ref = self._approx_log_reference(observed_target_uni,
                                                        cov_target_uni,
                                                        T[m],
                                                        eval_grid[m])
            
            if self.ncoarse is None:

                logW = (approx_log_ref - 0.5 * (self.stat_grid[m] - TS.observed_target[m]) ** 2 / var_target)
                logW -= logW.max()
                _log_ref[m,:] = logW
                self._families.append(discrete_family(self.stat_grid[m],
                                                      np.exp(logW)))
            else:

                approx_fn = interp1d(eval_grid[m],
                                     approx_log_ref,
                                     kind='quadratic',
                                     bounds_error=False,
                                     fill_value='extrapolate')

                grid = self.stat_grid[m]
                logW = (approx_fn(grid) -
                        0.5 * (grid - TS.observed_target[m]) ** 2 / var_target)

                logW -= logW.max()

                DEBUG = False # JT: this can be removed 
                if DEBUG:
                    approx_log_ref2 = self._approx_log_reference(observed_target_uni,
                                                                 cov_target_uni,
                                                                 T[m],
                                                                 grid)
                    logW2 = (approx_log_ref2 - 0.5 * (grid - TS.observed_target[m]) ** 2 / var_target)
                    logW2 -= logW2.max()
                    import matplotlib.pyplot as plt
                    plt.plot(grid, logW, label='extrapolated')

                    plt.plot(grid, logW2, label='fine grid')
                    plt.legend()

                    plt.figure(num=2)
                    plt.plot(eval_grid[m], approx_fn(eval_grid[m]), label='extrapolated coarse')
                    plt.plot(grid, approx_fn(grid), label='extrapolated fine')
                    plt.plot(grid, approx_log_ref2, label='fine grid')
                    plt.legend()

                    plt.show()
                    stop

                _log_ref[m, :] = logW
                self._families.append(discrete_family(grid,
                                                      np.exp(logW)))

        self._log_ref = _log_ref

