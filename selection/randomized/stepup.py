from __future__ import print_function
import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from ..constraints.affine import constraints
from .query import affine_gaussian_sampler
from .randomization import randomization
from .marginal_screening import marginal_screening
from ..algorithms.debiased_lasso import debiasing_matrix

def stepup_selection(Z_values, stepup_Z):

    m = Z_values.shape[0]
    absZ_sorted = np.sort(np.fabs(Z_values))[::-1]
    survivors = absZ_sorted - stepup_Z >= 0
    if np.any(survivors):
        last_survivor = np.max(np.nonzero(survivors)[0])
        return last_survivor + 1
    else:
        return 0

class stepup(marginal_screening):

    def __init__(self,
                 observed_score,
                 covariance, 
                 step_Z,
                 randomizer,
                 perturb=None):

        self.observed_score_state = observed_score
        self.nfeature = p = observed_score.shape[0]
        self.covariance = covariance
        self.step_Z = step_Z
        
        if not (np.all(sorted(self.step_Z)[::-1] == self.step_Z) and
                np.all(np.greater_equal(self.step_Z, 0))):
            raise ValueError('stepup Z values should be non-negative and non-increasing')

        self.randomizer = randomizer
        self._initial_omega = perturb

    def fit(self, perturb=None):

        # condition on all those that survive
        # and the observed (randomized) Z values of those that don't

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        _randomized_score = -self.observed_score_state + self._initial_omega

        K = stepup_selection(_randomized_score, self.step_Z)
        if K < p:
            Z_cutoff = self.step_Z[K] # or K \pm 1? check!
        else:
            Z_cutoff = 0

        self._selected = np.fabs(_randomized_score) > Z_cutoff
        self._not_selected = ~self._selected
        sign = np.sign(_randomized_score)
        active_signs = sign[self._selected]
        sign[self._not_selected] = 0
        self.selection_variable = {'sign': sign.copy(),
                                   'variables': self._selected.copy(),
                                   # also conditioning on values of randomized unselected coefficients
                                   }

        self.observed_opt_state = np.fabs(_randomized_score[self._selected]) - Z_cutoff 

        self.num_opt_var = self.observed_opt_state.shape[0]
        opt_linear = np.zeros((p, self.num_opt_var))
        opt_linear[self._selected,:] = np.identity(self.num_opt_var)
        opt_offset = np.zeros(p)
        opt_offset[self._selected] = active_signs * Z_cutoff
        opt_offset[self._not_selected] = _randomized_score[self._not_selected]

        self.opt_transform = (opt_linear, opt_offset)
        self._setup = True

        _, prec = self.randomizer.cov_prec

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            logdens_linear = cond_cov.dot(opt_linear.T) * prec  
        else:
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            logdens_linear = cond_cov.dot(opt_linear.T).dot(prec)

        cond_mean = logdens_linear.dot(-self.observed_score_state - opt_offset)
        logdens_transform = (logdens_linear, opt_offset)
        A_scaling = -np.identity(len(active_signs))
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

    @staticmethod
    def BH(observed_score,
           covariance, 
           randomizer_scale,
           q=0.2,
           perturb=None):
        
        # XXXX implicitly making assumption diagonal is constant -- seems reasonable for BH

        p = observed_score.shape[0]

        randomized_stdev = np.sqrt(np.diag(covariance).mean() + randomizer_scale**2)
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        stepup_Z = randomized_stdev * ndist.ppf(1 - q * np.arange(1, p + 1) / (2 * p))

        return stepup(observed_score,
                      covariance,
                      stepup_Z,
                      randomizer,
                      perturb=perturb)
