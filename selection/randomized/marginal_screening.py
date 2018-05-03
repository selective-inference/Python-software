from __future__ import print_function
import functools
import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from .base import restricted_estimator
from ..constraints.affine import constraints
from .query import affine_gaussian_sampler
from .randomization import randomization
from ..algorithms.debiased_lasso import debiasing_matrix

def marginal_screening_selection(p_values, level):

    m = p_values.shape[0]
    p_sorted = np.sort(p_values)
    indices = np.arange(m)
    indices_order = np.argsort(p_values)
    E_sel = (p_values <= level)
    nselect = E_sel.sum()
    not_sel = ~E_sel

    active = np.zeros(m, np.bool)
    active[E_sel] = 1

    return nselect, active, np.argsort(p_values[np.sort(not_sel)])

def BH_selection(p_values, level):

    m = p_values.shape[0]
    p_sorted = np.sort(p_values)
    indices = np.arange(m)
    indices_order = np.argsort(p_values)
    first_step = p_sorted - level * (np.arange(m) + 1.) / m <= 0
    if np.any(first_step):
        order_sig = np.max(indices[first_step])
        E_sel = indices_order[:(order_sig+1)]
        not_sel = indices_order[(order_sig+1):]

        active = np.zeros(m, np.bool)
        active[E_sel] = 1

        return order_sig + 1, active, np.argsort(p_values[np.sort(not_sel)])
    else:
        return 0, np.zeros(m, np.bool), np.argsort(p_values)

class marginal_screening(object):

    def __init__(self,
                 observed_score,
                 covariance, 
                 randomizer_scale,
                 marginal_level,
                 perturb=None):

        self.observed_score = observed_score  # - Z if Z \sim N(\mu,\Sigma)
        self.nfeature = p = observed_score.shape[0]
        self.covariance = covariance
        self.randomized_stdev = np.sqrt(np.diag(covariance) + randomizer_scale**2)
        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        self._initial_omega = perturb
        self.marginal_level = marginal_level

    def fit(self, perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        randomized_score = -self.observed_score + self._initial_omega
        p_values = 2. * (1. - ndist.cdf(np.abs(randomized_score) / self.randomized_stdev))
        K, active, sort_notsel_pvals = marginal_screening_selection(p_values, self.marginal_level)

        self.threshold = self.randomized_stdev * ndist.ppf(1. - self.marginal_level / 2.)
        self._selected = active
        self._not_selected = ~self._selected
        sign = np.sign(randomized_score)
        active_signs = sign[self._selected]
        sign[self._not_selected] = 0
        self.selection_variable = {'sign': sign,
                                   'variables': self._selected.copy()}

        self.observed_opt_state = (self._initial_omega[self._selected] - 
                                   self.observed_score[self._selected] - 
                                   np.diag(active_signs).dot(self.threshold[self._selected]))
        self.num_opt_var = self.observed_opt_state.shape[0]

        opt_linear = np.zeros((p, self.num_opt_var))
        opt_linear[self._selected,:] = np.diag(active_signs)
        opt_offset = np.zeros(p)
        opt_offset[self._selected] = active_signs * self.threshold[self._selected]
        opt_offset[self._not_selected] = randomized_score[self._not_selected]

        self.opt_transform = (opt_linear, opt_offset)

        cov, prec = self.randomizer.cov_prec
        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec
        cond_mean = -logdens_linear.dot(self.observed_score + opt_offset)

        logdens_transform = (logdens_linear, opt_offset)
        A_scaling = -np.diag(active_signs)
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
        return self._selected

    def summary(self,
                observed_target,
                cov_target,
                cov_target_score,
                alternatives,
                parameter=None,
                level=0.9,
                ndraw=10000,
                burnin=2000,
                compute_intervals=False):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        """

        if parameter is None:
            parameter = np.zeros(observed_target.shape[0])

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target,
                                                  cov_target,
                                                  cov_target_score,
                                                  parameter=parameter,
                                                  sample=opt_sample,
                                                  alternatives=alternatives)
        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target,
                                                       cov_target,
                                                       cov_target_score,
                                                       parameter=np.zeros_like(parameter),
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target,
                                                          cov_target,
                                                          cov_target_score,
                                                          sample=opt_sample)

        return pivots, pvalues, intervals

    def selective_MLE(self,
                      observed_target,
                      cov_target,
                      cov_target_score,
                      alternatives,
                      level=0.9,
                      solve_args={'tol':1.e-12}):
        """

        Parameters
        ----------

        level : float
            Confidence level.

        """

        return self.sampler.selective_MLE(observed_target,
                                          cov_target,
                                          cov_target_score,
                                          self.observed_opt_state,
                                          solve_args=solve_args)

    def form_targets(self, features):

        score_linear = -self.covariance[:,features]
        Q = -score_linear[features]
        cov_target = np.linalg.inv(Q)
        observed_target = np.linalg.inv(Q).dot(-self.observed_score[features])
        crosscov_target_score = score_linear.dot(cov_target)
        alternatives = ([{1: 'greater', -1: 'less'}[int(s)] for s in 
                         self.selection_variable['sign'][features]])

        return observed_target, cov_target, crosscov_target_score.T, alternatives

class BH(marginal_screening):

    def __init__(self,
                 observed_score,
                 covariance, 
                 randomizer_scale,
                 BH_level,
                 perturb=None):

        self.observed_score = observed_score
        self.nfeature = p = observed_score.shape[0]
        self.covariance = covariance
        self.randomized_stdev = np.sqrt(np.diag(covariance) + randomizer_scale**2)
        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        self._initial_omega = perturb

        self.BH_level = BH_level

    def fit(self, perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        randomized_score = -self.observed_score + self._initial_omega
        p_values = 2. * (1. - ndist.cdf(np.abs(randomized_score) / self.randomized_stdev))
        K, active, sort_notsel_pvals = BH_selection(p_values, self.BH_level)
        BH_cutoff = self.randomized_stdev * ndist.ppf(1. - (K * self.BH_level) /(2.*p))

        if np.array(BH_cutoff).shape in [(), (1,)]:
            BH_cutoff = np.ones(p) * BH_cutoff
        self.BH_cutoff = BH_cutoff

        self._selected = np.fabs(randomized_score) > self.BH_cutoff
        self._not_selected = ~self._selected
        sign = np.sign(randomized_score)
        active_signs = sign[self._selected]
        sign[self._not_selected] = 0
        self.selection_variable = {'sign': sign.copy(),
                                   'variables': self._selected.copy()}

        threshold = np.zeros(p)
        threshold[self._selected] = self.BH_cutoff[self._selected]
        cut_off_vector = ndist.ppf(1. - ((K + np.arange(self._not_selected.sum()) + 1)
                                         * self.BH_level) / float(2.* p))

        indices_interior = np.asarray([u for u in range(p) if self._not_selected[u]])
        threshold[indices_interior[sort_notsel_pvals]] = \
            (self.randomized_stdev[self._not_selected])[sort_notsel_pvals] * cut_off_vector

        self.threshold = threshold

        self.observed_opt_state = (self._initial_omega[self._selected] - 
                                   self.observed_score[self._selected] - 
                                  np.diag(active_signs).dot(self.threshold[self._selected]))
        self.num_opt_var = self.observed_opt_state.shape[0]

        opt_linear = np.zeros((p, self.num_opt_var))
        opt_linear[self._selected, :] = np.diag(active_signs)
        opt_offset = np.zeros(p)
        opt_offset[self._selected] = active_signs * self.threshold[self._selected]
        opt_offset[self._not_selected] = randomized_score[self._not_selected]

        self.opt_transform = (opt_linear, opt_offset)

        cov, prec = self.randomizer.cov_prec
        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec
        cond_mean = -logdens_linear.dot(self.observed_score + opt_offset)

        logdens_transform = (logdens_linear, opt_offset)
        print(active_signs)

        A_scaling = -np.diag(active_signs)
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
        return self._selected


