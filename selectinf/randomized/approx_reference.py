from __future__ import division, print_function
from typing import NamedTuple

import numpy as np, pandas as pd
from scipy.interpolate import interp1d

from ..distributions.discrete_family import discrete_family
from ..algorithms.barrier_affine import solve_barrier_affine_py
from .selective_MLE import mle_inference
from .base import target_query_Interactspec

class ConditionalSpec(NamedTuple):

    # description of (preselection) conditional law of
    # targets \hat{\theta} | u, N
    # if they were unbiased, then:
    # 1) precision will agree with marginal variance
    # 2) scalings will all be 1
    # 3) shifts will be 0

    precision : np.ndarray
    scalings : np.ndarray
    shifts : np.ndarray
    T : np.ndarray  # what is T?

class grid_inference(object):

    def __init__(self,
                 query_spec,
                 target_spec,
                 solve_args={'tol': 1.e-12},
                 ngrid=1000):

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

        self.query_spec = query_spec
        self.target_spec = target_spec
        self.solve_args = solve_args
        self.ngrid = ngrid

        G = mle_inference(query_spec,
                          target_spec,
                          solve_args=solve_args)

        _, inverse_info, log_ref = G.solve_estimating_eqn()

        TS = target_spec
        self.ntarget = ntarget = TS.cov_target.shape[0]
        _scale = 4 * np.sqrt(np.diag(inverse_info))
        self.inverse_info = inverse_info

        self.stat_grid = np.zeros((ntarget, ngrid))
        for j in range(ntarget):
            self.stat_grid[j, :] = np.linspace(TS.observed_target[j] - 1.5 * _scale[j],
                                               TS.observed_target[j] + 1.5 * _scale[j],
                                               num=ngrid)

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

        TS = self.target_spec

        if parameter is not None:
            pivots = self._pivots(parameter,
                                  alternatives=alternatives)
        else:
            pivots = None

        pvalues = self._pivots(np.zeros_like(TS.observed_target),
                                      alternatives=alternatives) 
        lower, upper = self._intervals(level=level)

        result = pd.DataFrame({'target': TS.observed_target,
                               'pvalue': pvalues,
                               'alternative': alternatives,
                               'lower_confidence': lower,
                               'upper_confidence': upper})

        if not np.all(parameter == 0):
            result.insert(4, 'pivot', pivots)
            result.insert(5, 'parameter', parameter)

        return result

    def _approx_log_reference(self,
                              observed_target,
                              cov_target,
                              linear_coef,
                              grid):

        """
        Approximate the log of the reference density on a grid.
        """
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

            cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) + self.cond_mean)
            conjugate_arg = self.cond_precision.dot(cond_mean_grid)

            val, _, _ = solver(conjugate_arg,
                               self.cond_precision,
                               self.observed_soln,
                               self.linear_part,
                               self.offset,
                               **self.solve_args)

            ref_hat.append(-val - (conjugate_arg.T.dot(self.cond_cov).dot(conjugate_arg) / 2.))

        return np.asarray(ref_hat)

    def _pivots(self,
                mean_parameter,
                alternatives=None):

        TS = self.target_spec
        
        if not hasattr(self, "_families"):
            self._construct_density() # generic
            self._construct_families() # specific to the method
        precs, S, r, _ = self.conditional_spec

        if alternatives is None:
            alternatives = ['twosided'] * self.ntarget

        pivot = []

        for m in range(self.ntarget):

            family = self._families[m]
            var_target = 1. / (precs[m][0, 0])

            mean = S[m].dot(mean_parameter[m].reshape((1,))) + r[m]
            # construction of pivot from families follows `selectinf.learning.core`

            _cdf = family.cdf((mean[0] - TS.observed_target[m]) / var_target, x=TS.observed_target[m])

            if alternatives[m] == 'twosided':
                pivot.append(2 * min(_cdf, 1 - _cdf))
            elif alternatives[m] == 'greater':
                pivot.append(1 - _cdf)
            elif alternatives[m] == 'less':
                pivot.append(_cdf)
            else:
                raise ValueError('alternative should be in ["twosided", "less", "greater"]')
        return pivot # , self._log_ref

    def _intervals(self,
                   level=0.9):

        TS = self.target_spec
        
        if not hasattr(self, "_families"):
            self._construct_density() # generic
            self._construct_families() # specific to the method

        precs, S, r, _ = self.conditional_spec

        lower, upper = [], []

        for m in range(self.ntarget):
            # construction of intervals from families follows `selectinf.learning.core`
            family = self._families[m]
            observed_target = TS.observed_target[m]
            unbiased_est = (observed_target - r[m][0]) * (1./(S[m][0,0]))

            _l, _u = family.equal_tailed_interval(observed_target,
                                                  alpha=1 - level)
            l = _l * (1./(S[m][0,0]))
            u = _u * (1./(S[m][0,0]))

            var_target = 1. / (precs[m][0, 0])

            lower.append(l * var_target + unbiased_est)
            upper.append(u * var_target + unbiased_est)

        return np.asarray(lower), np.asarray(upper)

    ### Private method

    def _construct_density(self):
        """
        What is this method doing?
        """

        TS = self.target_spec
        QS = self.query_spec

        precs = []
        S = []
        r = []
        T = []

        p = TS.regress_target_score.shape[1]

        for m in range(self.ntarget):
            observed_target_uni = (TS.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(TS.cov_target)[m]).reshape((1, 1))
            regress_target_score_uni = TS.regress_target_score[m, :].reshape((1, p))

            U1, U2, U3, U4, U5 = self._form_interaction_pieces(QS,
                                                               regress_target_score_uni,
                                                               cov_target_uni)

            prec_target = 1. / cov_target_uni

            # JT: what is _T?
            _T = QS.cond_cov.dot(U5.T)

            prec_target_nosel = prec_target + U2 - U3

            _P = -(U1.T.dot(QS.M5) + U2.dot(observed_target_uni))

            bias_target = cov_target_uni.dot(
                U1.T.dot(-U4.dot(observed_target_uni) + QS.M4.dot(QS.cond_mean)) - _P)

            _r = np.linalg.inv(prec_target_nosel).dot(prec_target.dot(bias_target))
            _S = np.linalg.inv(prec_target_nosel).dot(prec_target)

            S.append(_S)
            r.append(_r)
            precs.append(prec_target_nosel)
            T.append(_T)

        self.conditional_spec = ConditionalSpec(np.array(precs),
                                                np.array(S),
                                                np.array(r),
                                                np.array(T) # what is T here?
                                                )

        return self.conditional_spec

    # Private

    def _form_interaction_pieces(self,
                                 QS,
                                 regress_target_score,
                                 cov_target):

        return target_query_Interactspec(QS,
                                         regress_target_score,
                                         cov_target)
    

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

            cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) +
                              QS.cond_mean)
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

