from __future__ import print_function
import functools
import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from .query import gaussian_query
from .randomization import randomization
from ..base import TargetSpec

class screening(gaussian_query):

    def __init__(self,
                 observed_data,
                 covariance, # unscaled
                 randomizer,
                 perturb=None):

        self.observed_score_state = -observed_data  # -Z if Z \sim N(\mu,\Sigma), X^Ty in regression setting
        self.nfeature = p = self.observed_score_state.shape[0]
        self.covariance = covariance
        self.randomizer = randomizer
        self._initial_omega = perturb
        self._unscaled_cov_score = covariance

    def fit(self, perturb=None):

        gaussian_query.fit(self, perturb=perturb)
        self._randomized_score = self.observed_score_state - self._initial_omega
        return self._randomized_score, self._randomized_score.shape[0]

    def multivariate_targets(self,
                             features,
                             dispersion=1):
        """
        Entries of the mean of \Sigma[E,E]^{-1}Z_E
        """
        Q = self.covariance[features][:,features] 
        Qinv = np.linalg.inv(Q)
        cov_target = np.linalg.inv(Q) * dispersion
        observed_target = -np.linalg.inv(Q).dot(self.observed_score_state[features])
        regress_target_score = -Qinv.dot(np.identity(self.covariance.shape[0])[features])
        alternatives = ['twosided'] * features.sum()

        return TargetSpec(observed_target, 
                          cov_target,
                          regress_target_score,
                          alternatives,
                          dispersion)

    def full_targets(self,
                     features,
                     dispersion=1):
        """
        Entries of the mean of (\Sigma^{-1}Z)[E]
        """

        Q = self.covariance
        Qinv = np.linalg.inv(Q)
        cov_target = Qinv[features][:, features] * dispersion
        observed_target = -np.linalg.inv(Q).dot(self.observed_score_state)[features]
        regress_target_score = -Qinv[:, features]
        alternatives = ['twosided'] * features.sum()

        return TargetSpec(observed_target,
                          cov_target,
                          regress_target_score.T,
                          alternatives,
                          dispersion)

    def marginal_targets(self,
                         features,
                         dispersion=1):
        """
        Entries of the mean of Z_E
        """
        Q = self.covariance[features][:,features] 
        cov_target = Q * dispersion
        observed_target = -self.observed_score_state[features]
        regress_target_score = -np.identity(self.covariance.shape[0])[:,features]
        alternatives = ['twosided'] * features.sum()

        return TargetSpec(observed_target,
                          cov_target,
                          regress_target_score.T,
                          alternatives,
                          dispersion)

class marginal_screening(screening):

    def __init__(self,
                 observed_data,
                 covariance, 
                 randomizer,
                 threshold,
                 perturb=None):

        threshold = np.asarray(threshold)
        if threshold.shape == ():
            threshold = np.ones_like(observed_data) * threshold

        self.threshold = threshold
        screening.__init__(self,
                           observed_data,
                           covariance, 
                           randomizer,
                           perturb=None)                           

    def fit(self, perturb=None):

        _randomized_score, p = screening.fit(self, perturb=perturb)
        active = np.fabs(_randomized_score) >= self.threshold

        self._selected = active
        self._not_selected = ~self._selected
        sign = np.sign(-_randomized_score)
        active_signs = sign[self._selected]
        sign[self._not_selected] = 0
        self.selection_variable = {'sign': sign,
                                   'variables': self._selected.copy()}

        self.observed_opt_state = (np.fabs(_randomized_score) - self.threshold)[self._selected]
        self.num_opt_var = self.observed_opt_state.shape[0]

        opt_linear = np.zeros((p, self.num_opt_var))
        opt_linear[self._selected] = np.diag(active_signs)
        observed_subgrad = np.zeros(p)
        observed_subgrad[self._selected] = active_signs * self.threshold[self._selected]
        observed_subgrad[self._not_selected] = _randomized_score[self._not_selected]

        self._setup = True

        A_scaling = -np.identity(len(active_signs))
        b_scaling = np.zeros(self.num_opt_var)

        self._setup_sampler(A_scaling,
                            b_scaling,
                            opt_linear,
                            observed_subgrad)

        return self._selected

    @staticmethod
    def type1(observed_data,
              covariance, 
              marginal_level,
              randomizer_scale,
              perturb=None):
        '''
        Threshold
        '''

        randomized_stdev = np.sqrt(np.diag(covariance) + randomizer_scale**2)
        p = covariance.shape[0]
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        threshold = randomized_stdev * ndist.ppf(1. - marginal_level / 2.)

        return marginal_screening(observed_data,
                                  covariance, 
                                  randomizer,
                                  threshold,
                                  perturb=perturb)

# Stepup procedures like Benjamini-Hochberg

def stepup_selection(Z_values, stepup_Z):

    absZ_argsort = np.argsort(np.fabs(Z_values))[::-1]
    absZ_sorted = np.fabs(Z_values)[absZ_argsort]
    survivors = absZ_sorted - stepup_Z >= 0
    if np.any(survivors):
        num_selected = max(np.nonzero(survivors)[0]) + 1
        return (num_selected,                    # how many selected
                absZ_argsort[:num_selected],     # ordered indices of those selected
                stepup_Z[num_selected - 1])      # the selected are greater than this number 
    else:
        return 0, None, None

class stepup(screening):

    def __init__(self,
                 observed_data,
                 covariance, 
                 randomizer,
                 stepup_Z,
                 perturb=None):

        screening.__init__(self,
                           observed_data,
                           covariance, 
                           randomizer,
                           perturb=None)                           

        self.stepup_Z = stepup_Z
        if not (np.all(sorted(self.stepup_Z)[::-1] == self.stepup_Z) and
                np.all(np.greater_equal(self.stepup_Z, 0))):
            raise ValueError('stepup Z values should be non-negative and non-increasing')

    def fit(self, perturb=None):

        # condition on all those that survive, their sign,
        # which was the last past the threshold
        # and the observed (randomized) Z values of those that don't

        _randomized_score, p = screening.fit(self, perturb=perturb)

        K, selected_idx, last_cutoff = stepup_selection(_randomized_score, self.stepup_Z)

        if K > 0:

            self._selected = np.zeros(p, np.bool)
            self._selected[selected_idx] = 1
            self._not_selected = ~self._selected

            sign = np.sign(-_randomized_score)
            active_signs = sign[selected_idx]
            sign[self._not_selected] = 0
            self.selection_variable = {'sign': sign.copy(),
                                       'variables': self._selected.copy(),
                                       }

            self.num_opt_var = self._selected.sum()
            self.observed_opt_state = np.zeros(self.num_opt_var)
            self.observed_opt_state[:] = np.fabs(_randomized_score[selected_idx]) - last_cutoff

            opt_linear = np.zeros((p, self.num_opt_var))
            for j in range(self.num_opt_var):
                opt_linear[selected_idx[j], j] = active_signs[j]

            observed_subgrad = np.zeros(p)
            observed_subgrad[self._selected] = active_signs * last_cutoff
            observed_subgrad[self._not_selected] = _randomized_score[self._not_selected]

            self._setup = True

            A_scaling = -np.identity(self.num_opt_var)
            b_scaling = np.zeros(self.num_opt_var)

            self._setup_sampler(A_scaling,
                                b_scaling,
                                opt_linear,
                                observed_subgrad)
        else:
            self._selected = np.zeros(p, np.bool)
        return self._selected


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
                      randomizer,
                      stepup_Z,
                      perturb=perturb)

    def BH_general(observed_score,
                   covariance, 
                   randomizer_cov,
                   q=0.2,
                   perturb=None):
        
        if not np.allclose(np.diag(covariance), covariance[0, 0]):
            raise ValueError('Benjamin-Hochberg expecting Z scores with identical variance, standardize your Z')
        if not np.allclose(np.diag(randomizer_cov), randomizer_cov[0, 0]):
            raise ValueError('Benjamin-Hochberg expecting Z scores with identical variance, standardize your randomization')

        p = observed_score.shape[0]

        randomizer = randomization.gaussian(randomizer_cov)
        randomized_stdev = np.sqrt(covariance[0, 0] + randomizer_cov[0, 0])

        # Benjamini-Hochberg cutoffs
        stepup_Z = randomized_stdev * ndist.ppf(1 - q * np.arange(1, p + 1) / (2 * p))

        return stepup(observed_score,
                      covariance,
                      randomizer,
                      stepup_Z,
                      perturb=perturb)

class topK(screening):

    def __init__(self,
                 observed_data,
                 covariance, 
                 randomizer,
                 K,         # how many to select?
                 abs=False, # absolute value of scores or not?
                 perturb=None):

        screening.__init__(self,
                           observed_data,
                           covariance, 
                           randomizer,
                           perturb=None)                           

        self.K = K
        self._abs = abs

    def fit(self, perturb=None):

        _randomized_score, p = screening.fit(self, perturb=perturb)

        # fixing the topK
        # gives us that u=\omega - Z is in a particular cone
        # like: s_i u_i \geq |u_j|  1 \leq i \leq K, K+1 \leq j \leq p (when abs is True and s_i are topK signs) or
        # u_i \geq u_j  1 \leq i \leq K, K+1 \leq j \leq p (when abs is False)

        if self._abs:

            Z = np.fabs(-_randomized_score)
            topK = np.argsort(Z)[-self.K:]

            selected = np.zeros(self.nfeature, np.bool)
            selected[topK] = 1
            self._selected = selected
            self._not_selected = ~self._selected

            sign = np.sign(Z)
            topK_signs = sign[self._selected]
            sign[self._not_selected] = 0

            self.selection_variable = {'sign': sign,
                                       'variables': self._selected.copy()}

            self.observed_opt_state = np.fabs(Z[self._selected])
            self.num_opt_var = self.observed_opt_state.shape[0]

            opt_linear = np.zeros((p, self.num_opt_var))
            opt_linear[self._selected] = np.diag(topK_signs)
            observed_subgrad = np.zeros(p)  

        else:

            Z = -_randomized_score
            topK = np.argsort(Z)[-self.K:]

            selected = np.zeros(self.nfeature, np.bool)
            selected[topK] = 1
            self._selected = selected
            self._not_selected = ~self._selected
            self.selection_variable = {'variables': self._selected.copy()}

            self.observed_opt_state = Z[self._selected]
            self.num_opt_var = self.observed_opt_state.shape[0]

            opt_linear = np.zeros((p, self.num_opt_var))
            opt_linear[self._selected] = np.identity(self.num_opt_var)
            observed_subgrad = np.zeros(p)  

        # in both cases, this conditioning means we just need to compute
        # the observed lower bound
        # XXX what about abs?

        lower_bound = np.max(Z[self._not_selected])

        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = -np.ones(self.num_opt_var) * lower_bound

        self._setup_sampler(A_scaling,
                            b_scaling,
                            opt_linear,
                            observed_subgrad)

        return self._selected


