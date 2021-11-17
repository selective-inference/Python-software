import numpy as np, pandas as pd

from .selective_MLE import mle_inference

class grid_inference(object):

    def __init__(self,
                 query_spec,
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

        self.query_spec = query_spec
        self.target_spec = target_spec
        self.solve_args = solve_args

        G = mle_inference(query_spec,
                          target_spec,
                          solve_args=solve_args)

        _, inverse_info, log_ref = G.solve_estimating_eqn()

        TS = target_spec
        self.ntarget = ntarget = TS.cov_target.shape[0]
        _scale = 4 * np.sqrt(np.diag(inverse_info))
        self.inverse_info = inverse_info

        ngrid = 1000
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

    def _construct_families(self):

        self._construct_density()

        self._families = []
        _log_ref = np.zeros((self.ntarget, 1000))
        for m in range(self.ntarget):

            observed_target_uni = (self.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(self.cov_target)[m]).reshape((1, 1))

            var_target = 1. / ((self.precs[m])[0, 0])

            approx_log_ref = self._approx_log_reference(observed_target_uni,
                                                        cov_target_uni,
                                                        self.T[m],
                                                        self.stat_grid[m])

            if self.useIP == False:

                logW = (approx_log_ref - 0.5 * (self.stat_grid[m] - self.observed_target[m]) ** 2 / var_target)
                logW -= logW.max()
                _log_ref[m,:] = logW
                self._families.append(discrete_family(self.stat_grid[m],
                                                      np.exp(logW)))
            else:

                approx_fn = interp1d(self.stat_grid[m],
                                     approx_log_ref,
                                     kind='quadratic',
                                     bounds_error=False,
                                     fill_value='extrapolate')

                grid = np.linspace(self.stat_grid[m].min(), self.stat_grid[m].max(), 1000)
                logW = (approx_fn(grid) -
                        0.5 * (grid - self.observed_target[m]) ** 2 / var_target)

                logW -= logW.max()
                _log_ref[m, :] = logW
                self._families.append(discrete_family(grid,
                                                      np.exp(logW)))

        self._log_ref = _log_ref

    def _pivots(self,
                mean_parameter,
                alternatives=None):

        TS = self.target_spec

        if not hasattr(self, "_families"):
            self._construct_families()

        if alternatives is None:
            alternatives = ['twosided'] * self.ntarget

        pivot = []

        for m in range(self.ntarget):

            family = self._families[m]
            var_target = 1. / ((self.precs[m])[0, 0])

            mean = self.S[m].dot(mean_parameter[m].reshape((1,))) + self.r[m]
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
            self._construct_families()

        lower, upper = [], []

        for m in range(self.ntarget):
            # construction of intervals from families follows `selectinf.learning.core`
            family = self._families[m]
            observed_target = TS.observed_target[m]

            l, u = family.equal_tailed_interval(observed_target,
                                                alpha=1 - level)

            var_target = 1. / ((self.precs[m])[0, 0])

            lower.append(l * var_target + observed_target)
            upper.append(u * var_target + observed_target)

        return np.asarray(lower), np.asarray(upper)

    ### Private method
    def _construct_density(self):

        TS = self.target_spec
        QS = self.query_spec

        precs = {}
        S = {}
        r = {}
        T = {}

        p = TS.regress_target_score.shape[1]

        for m in range(self.ntarget):
            observed_target_uni = (TS.observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(TS.cov_target)[m]).reshape((1, 1))
            prec_target = 1. / cov_target_uni
            regress_target_score_uni = TS.regress_target_score[m, :].reshape((1, p))

            T1 = regress_target_score_uni.T.dot(prec_target)
            T2 = T1.T.dot(QS.M2.dot(T1))
            T3 = T1.T.dot(QS.M3.dot(T1))
            T4 = QS.M1.dot(QS.opt_linear).dot(QS.cond_cov).dot(QS.opt_linear.T.dot(QS.M1.T.dot(T1)))
            T5 = T1.T.dot(QS.M1.dot(QS.opt_linear))

            _T = QS.cond_cov.dot(T5.T)

            prec_target_nosel = prec_target + T2 - T3

            _P = -(T1.T.dot(QS.M1.dot(QS.observed_score)) + T2.dot(observed_target_uni))

            bias_target = cov_target_uni.dot(
                T1.T.dot(-T4.dot(observed_target_uni) + QS.M1.dot(QS.opt_linear.dot(QS.cond_mean))) - _P)

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
