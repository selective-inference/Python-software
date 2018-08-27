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
    absZ_argsort = np.argsort(np.fabs(Z_values))[::-1]
    absZ_sorted = np.fabs(Z_values)[absZ_argsort]
    survivors = absZ_sorted - stepup_Z >= 0
    if np.any(survivors):
        num_selected = np.max(np.nonzero(survivors)[0]) + 1
        return (num_selected,                    # how many selected
                absZ_argsort[:num_selected],     # ordered indices of those selected
                max(stepup_Z[num_selected - 1],  # the last selected is greater than this number 
                    absZ_sorted[num_selected]))  # conditional on the unselected ones
    else:
        return 0, None, None

class stepup(marginal_screening):

    def __init__(self,
                 observed_data,
                 covariance, 
                 step_Z,
                 randomizer,
                 perturb=None):

        self.observed_score_state = -observed_data  # -Z if Z \sim N(\mu,\Sigma), X^Ty in regression setting
        self.nfeature = p = observed_data.shape[0]
        self.covariance = covariance
        self.step_Z = step_Z
        
        if not (np.all(sorted(self.step_Z)[::-1] == self.step_Z) and
                np.all(np.greater_equal(self.step_Z, 0))):
            raise ValueError('stepup Z values should be non-negative and non-increasing')

        self.randomizer = randomizer
        self._initial_omega = perturb

    def fit(self, perturb=None):

        # condition on all those that survive, their sign,
        # which was the last past the threshold
        # and the observed (randomized) Z values of those that don't

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        _randomized_score = self.observed_score_state - self._initial_omega

        K, selected_idx, last_cutoff = stepup_selection(_randomized_score, self.step_Z)

        if K > 0:

            _selected_first = np.zeros(p, np.bool)
            _selected_first[selected_idx[:-1]] = 1
            _selected_first = np.nonzero(_selected_first)[0]
            _selected_last = selected_idx[-1]

            self._selected = np.zeros(p, np.bool)
            self._selected[selected_idx] = 1
            self._not_selected = ~self._selected

            sign = np.sign(-_randomized_score)
            active_signs_first = sign[_selected_first]
            active_sign_last = sign[_selected_last]
            active_signs = sign[self._selected]
            sign[self._not_selected] = 0
            self.selection_variable = {'sign': sign.copy(),
                                       'variables': self._selected.copy(),
                                       'last':_selected_last,
                                       # also conditioning on values of randomized unselected coefficients
                                       }

            self.num_opt_var = self._selected.sum()
            self.observed_opt_state = np.zeros(self.num_opt_var)
            self.observed_opt_state[:-1] = np.fabs(_randomized_score[_selected_first]) - last_cutoff
            self.observed_opt_state[-1] = np.fabs(_randomized_score[_selected_last]) - last_cutoff

            opt_linear = np.zeros((p, self.num_opt_var))
            for j in range(self.num_opt_var - 1):
                opt_linear[_selected_first[j], j] = active_signs_first[j]
            opt_linear[_selected_last,-1] = active_sign_last

            opt_offset = np.zeros(p)
            opt_offset[self._selected] = active_signs * last_cutoff
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

            # identify which is the smallest
            # and that it is non-negative
            A_scaling = []
            for i in range(self.num_opt_var - 1):
                row = np.zeros(self.num_opt_var)
                row[-1] = 1
                row[i] = -1.
                A_scaling.append(row)
            last_row = np.zeros(self.num_opt_var)
            last_row[-1] = -1
            A_scaling.append(last_row)
            A_scaling = np.array(A_scaling)

            b_scaling = np.zeros(self.num_opt_var)

            if not np.all(A_scaling.dot(self.observed_opt_state) - b_scaling <= 0):
                raise ValueError('constraints not satisfied')

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
        else:
            return None

    @staticmethod
    def BH(observed_score,
           covariance, 
           randomizer_scale,
           q=0.2,
           perturb=None):
        
        if not np.allclose(np.diag(covariance), covariance[0, 0]):
            raise ValueError('Benjamin-Hochberg expecting Z scores with identical variance, standardize your Z')

        p = observed_score.shape[0]

        randomized_stdev = np.sqrt(covariance[0, 0] + randomizer_scale**2)
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        # Benjamini-Hochberg cutoffs
        stepup_Z = randomized_stdev * ndist.ppf(1 - q * np.arange(1, p + 1) / (2 * p))

        return stepup(observed_score,
                      covariance,
                      stepup_Z,
                      randomizer,
                      perturb=perturb)
