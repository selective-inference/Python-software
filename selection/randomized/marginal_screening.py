from __future__ import print_function
import functools
import numpy as np
from selection.randomized.randomization import randomization
import regreg.api as rr
from selection.randomized.base import restricted_estimator
from selection.constraints.affine import constraints
from selection.randomized.query import (query,
                                        multiple_queries,
                                        langevin_sampler,
                                        affine_gaussian_sampler)
from scipy.stats import norm as ndist

from ..algorithms.debiased_lasso import debiasing_matrix

def BH_selection(p_values, level):

    m = p_values.shape[0]
    p_sorted = np.sort(p_values)
    indices = np.arange(m)
    indices_order = np.argsort(p_values)
    order_sig = np.max(indices[p_sorted - level * (np.arange(m) + 1.) / m <= 0])
    E_sel = indices_order[:(order_sig+1)]
    not_sel =indices_order[(order_sig+1):]

    active = np.zeros(m, np.bool)
    active[E_sel] = 1

    return order_sig+1, active, np.argsort(p_values[np.sort(not_sel)])

class BH(object):

    def __init__(self,
                 X,
                 Y,
                 sigma_hat,
                 randomizer_scale,
                 level,
                 perturb=None):

        observed_score = -X.T.dot(Y)
        self.nfeature = p = observed_score.shape[0]
        self.sigma_hat = sigma_hat

        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        self._initial_omega = perturb
        self.observed_score = observed_score

        self.level = level
        self.data = (X, Y)

    def fit(self, perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        randomized_score = -self.observed_score + self._initial_omega
        p_values = 2. * (1. - ndist.cdf(np.true_divide(np.abs(randomized_score),self.sigma_hat)))
        K, active, sort_notsel_pvals = BH_selection(p_values, self.level)
        BH_cutoff = self.sigma_hat * ndist.ppf(1. - (K * self.level) /(2.*p))
        if np.array(BH_cutoff).shape in [(), (1,)]:
            BH_cutoff = np.ones(p) * BH_cutoff
        self.BH_cutoff = BH_cutoff

        self.boundary = np.fabs(randomized_score) > self.BH_cutoff
        self.interior = ~self.boundary
        active_signs = np.sign(randomized_score)

        self.selection_variable = {'sign': active_signs.copy(),
                                   'variables': self.boundary.copy()}

        threshold = np.zeros(p)
        threshold[self.boundary] = self.BH_cutoff[self.boundary]
        cut_off_vector = ndist.ppf(1. - ((K+np.arange(self.interior.sum())+1) * self.level)/float(2.* p))

        indices_interior = np.asarray([u for u in range(p) if self.interior[u]])
        threshold[indices_interior[sort_notsel_pvals]] = (self.sigma_hat[self.interior])[sort_notsel_pvals] * cut_off_vector

        self.threshold = threshold

        self.observed_opt_state = self._initial_omega[self.boundary] - self.observed_score[self.boundary] - \
                                  np.diag(active_signs[self.boundary]).dot(self.threshold[self.boundary])
        self.num_opt_var = self.observed_opt_state.shape[0]

        opt_linear = np.zeros((p, self.num_opt_var))
        opt_linear[self.boundary, :] = np.diag(active_signs[self.boundary])
        opt_offset = np.zeros(p)
        opt_offset[self.boundary] = active_signs[self.boundary] * self.threshold[self.boundary]
        opt_offset[self.interior] = randomized_score[self.interior]

        print("check", (np.abs(opt_offset[self.interior])< threshold[self.interior]).sum(), self.interior.sum())
        self.opt_transform = (opt_linear, opt_offset)

        cov, prec = self.randomizer.cov_prec
        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec
        cond_mean = -logdens_linear.dot(self.observed_score + opt_offset)

        logdens_transform = (logdens_linear, opt_offset)
        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        def log_density(logdens_linear, offset, cond_prec, score, opt):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

        log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

        affine_con = constraints(A_scaling,
                                 b_scaling,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        self.sampler = affine_gaussian_sampler(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score,
                                               log_density,
                                               logdens_transform,
                                               selection_info=self.selection_variable)
        return self.boundary


    def selective_MLE(self,
                      target="selected",
                      features=None,
                      parameter=None,
                      level=0.9,
                      compute_intervals=False,
                      dispersion=None,
                      solve_args={'tol': 1.e-12}):
        """
        Parameters
        ----------
        target : one of ['selected', 'full']
        features : np.bool
            Binary encoding of which features to use in final
            model and targets.
        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.
        level : float
            Confidence level.
        ndraw : int (optional)
            Defaults to 1000.
        burnin : int (optional)
            Defaults to 1000.
        compute_intervals : bool
            Compute confidence intervals?
        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?
        """

        if parameter is None:
            parameter = np.zeros(self.nfeature)

        if target == 'selected':
            observed_target, cov_target, cov_target_score, alternatives = self.selected_targets(features=features,
                                                                                                dispersion=dispersion)

        elif target == 'full':
            X, y = self.data
            n, p = X.shape
            if n > p:
                observed_target, cov_target, cov_target_score, alternatives = self.full_targets(features=features,
                                                                                                dispersion=dispersion)
            else:
                observed_target, cov_target, cov_target_score, alternatives = self.debiased_targets(features=features,
                                                                                                    dispersion=dispersion)


        return self.sampler.selective_MLE(observed_target,
                                          cov_target,
                                          cov_target_score,
                                          self.observed_opt_state,
                                          solve_args=solve_args)

    def selected_targets(self, features=None, dispersion=None):

        X, y = self.data
        n, p = X.shape

        overall = self.boundary
        score_linear = -X.T.dot(X[:, overall])
        Q = -score_linear[overall]
        cov_target = np.linalg.inv(Q)
        observed_target = np.linalg.inv(Q).dot(X[:, overall].T.dot(y))
        crosscov_target_score = score_linear.dot(cov_target)
        alternatives = ([{1: 'greater', -1: 'less'}[int(s)] for s in self.selection_variable['sign'][self.boundary]])

        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    # def full_targets(self, features=None, dispersion=None):
    #
    #     if features is None:
    #         features = self.boundary
    #     features_bool = np.zeros(self.boundary.shape, np.bool)
    #     features_bool[features] = True
    #     features = features_bool
    #
    #     X, y = self.data
    #     n, p = X.shape
    #
    #     # target is one-step estimator
    #
    #     Qfull = X.T.dot(self._W[:, None] * X)
    #     G = self.loglike.smooth_objective(self.initial_soln, 'grad')
    #     Qfull_inv = np.linalg.inv(Qfull)
    #     one_step = self.initial_soln - Qfull_inv.dot(G)
    #     cov_target = Qfull_inv[features][:, features]
    #     observed_target = one_step[features]
    #     crosscov_target_score = np.zeros((p, cov_target.shape[0]))
    #     crosscov_target_score[features] = -np.identity(cov_target.shape[0])
    #
    #     if dispersion is None:  # use Pearson's X^2
    #         dispersion = ((y - self.loglike.saturated_loss.mean_function(X.dot(one_step))) ** 2 / self._W).sum() / (
    #         n - p)
    #
    #     alternatives = ['twosided'] * features.sum()
    #     return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives
    #
    # def debiased_targets(self,
    #                      features=None,
    #                      dispersion=None,
    #                      debiasing_args={}):
    #
    #     if features is None:
    #         features = self._overall
    #     features_bool = np.zeros(self._overall.shape, np.bool)
    #     features_bool[features] = True
    #     features = features_bool
    #
    #     X, y = self.data
    #     n, p = X.shape
    #
    #     # target is one-step estimator
    #
    #     G = self.loglike.smooth_objective(self.initial_soln, 'grad')
    #     Qinv_hat = np.atleast_2d(debiasing_matrix(X * np.sqrt(self._W)[:, None],
    #                                               np.nonzero(features)[0],
    #                                               **debiasing_args)) / n
    #     observed_target = self.initial_soln[features] - Qinv_hat.dot(G)
    #     if p > n:
    #         M1 = Qinv_hat.dot(X.T)
    #         cov_target = (M1 * self._W[None, :]).dot(M1.T)
    #         crosscov_target_score = -(M1 * self._W[None, :]).dot(X).T
    #     else:
    #         Qfull = X.T.dot(self._W[:, None] * X)
    #         cov_target = Qinv_hat.dot(Qfull.dot(Qinv_hat.T))
    #         crosscov_target_score = -Qinv_hat.dot(Qfull).T
    #
    #     if dispersion is None:  # use Pearson's X^2
    #         Xfeat = X[:, features]
    #         Qrelax = Xfeat.T.dot(self._W[:, None] * Xfeat)
    #         relaxed_soln = self.initial_soln[features] - np.linalg.inv(Qrelax).dot(G[features])
    #         dispersion = ((y - self.loglike.saturated_loss.mean_function(
    #             Xfeat.dot(relaxed_soln))) ** 2 / self._W).sum() / (n - features.sum())
    #
    #     alternatives = ['twosided'] * features.sum()
    #     return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    @staticmethod
    def gaussian(X,
                 Y,
                 sigma = 1.,
                 level = 0.10,
                 randomizer_scale=None):

        n, p = X.shape
        mean_diag = np.mean((X ** 2).sum(0))

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        sigma_hat = np.sqrt((sigma ** 2.) * (np.diag(X.T.dot(X))) + (randomizer_scale**2.))

        return BH(X, Y, sigma_hat, randomizer_scale, level)




