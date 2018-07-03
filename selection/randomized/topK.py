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

class topK(marginal_screening):

    def __init__(self,
                 observed_data,
                 covariance, 
                 randomizer_scale,
                 K,
                 abs=False, # absolute value of scores or not?
                 perturb=None):

        self.observed_score_state = -observed_data  # -Z if Z \sim N(\mu,\Sigma), X^Ty in regression setting
        self.nfeature = p = self.observed_score_state.shape[0]
        self.covariance = covariance
        randomized_stdev = np.sqrt(np.diag(covariance) + randomizer_scale**2)
        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        self._initial_omega = perturb
        self.K = K
        self._abs = abs

    def fit(self, perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        _randomized_score = self.observed_score_state - self._initial_omega

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
            opt_linear[self._selected,:] = np.diag(topK_signs)
            opt_offset = np.zeros(p)  
            self.opt_transform = (opt_linear, opt_offset)

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
            opt_linear[self._selected,:] = np.identity(self.num_opt_var)
            opt_offset = np.zeros(p)  
            self.opt_transform = (opt_linear, opt_offset)

        # in both cases, this conditioning means we just need to compute
        # the observed lower bound

        lower_bound = np.max(Z[self._not_selected])

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
        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = -np.ones(self.num_opt_var) * lower_bound

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




