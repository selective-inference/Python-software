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

class BH(marginal_screening):

    def __init__(self,
                 observed_score,
                 covariance, 
                 randomizer_scale,
                 BH_level,
                 perturb=None):

        self.observed_score_state = observed_score
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

        self._randomized_score = -self.observed_score_state + self._initial_omega
        p_values = 2. * (1. - ndist.cdf(np.abs(self._randomized_score) / self.randomized_stdev))
        K, active, sort_notsel_pvals = BH_selection(p_values, self.BH_level)
        BH_cutoff = self.randomized_stdev * ndist.ppf(1. - (K * self.BH_level) /(2.*p))

        if np.array(BH_cutoff).shape in [(), (1,)]:
            BH_cutoff = np.ones(p) * BH_cutoff
        self.BH_cutoff = BH_cutoff

        self._selected = np.fabs(self._randomized_score) > self.BH_cutoff
        self._not_selected = ~self._selected
        sign = np.sign(self._randomized_score)
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
                                   self.observed_score_state[self._selected] - 
                                  np.diag(active_signs).dot(self.threshold[self._selected]))
        self.num_opt_var = self.observed_opt_state.shape[0]

        opt_linear = np.zeros((p, self.num_opt_var))
        opt_linear[self._selected,:] = np.identity(self.num_opt_var)
        opt_offset = np.zeros(p)
        opt_offset[self._selected] = active_signs * self.threshold[self._selected]
        opt_offset[self._not_selected] = self._randomized_score[self._not_selected]

        self.opt_transform = (opt_linear, opt_offset)

        cov, prec = self.randomizer.cov_prec
        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec
        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

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
                                               self.observed_score_state,
                                               log_density,
                                               logdens_transform,
                                               selection_info=self.selection_variable)
        return self._selected


