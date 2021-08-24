from __future__ import division, print_function

import numpy as np
import nose.tools as nt

from scipy.stats import norm as ndist

import regreg.api as rr

from ..lasso import split_lasso 
from ...base import (selected_targets, 
                     full_targets, 
                     debiased_targets)                     
from ...tests.instance import gaussian_instance
from ...tests.flags import SET_SEED
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from ...algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
from ..randomization import randomization
from ...tests.decorators import rpy_test_safe

def test_split_lasso(n=100, 
                     p=200, 
                     signal_fac=3, 
                     s=5, 
                     sigma=3, 
                     target='full',
                     rho=0.4, 
                     proportion=0.5,
                     orthogonal=False,
                     ndraw=10000, 
                     MLE=True,
                     burnin=5000):
    """
    Test data splitting lasso
    """

    inst, const = gaussian_instance, split_lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    if orthogonal:
        X = np.linalg.svd(X, full_matrices=False)[0] * np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n) * sigma

    n, p = X.shape

    sigma_ = np.std(Y)
    W = np.ones(X.shape[1]) * np.sqrt(np.log(p)) * sigma_
    W[0] = 0

    conv = const(X, 
                 Y, 
                 W, 
                 proportion)
    
    signs = conv.fit()
    nonzero = signs != 0

    if nonzero.sum() > 0:

        if target == 'full':
            target_spec = full_targets(conv.loglike, 
                                       conv.observed_soln,
                                       dispersion=sigma**2)
        elif target == 'selected':
            target_spec = selected_targets(conv.loglike, 
                                           conv.observed_soln,
                                           dispersion=sigma**2)

        elif target == 'debiased':
            target_spec = debiased_targets(conv.loglike, 
                                           conv.observed_soln,
                                           penalty=conv.penalty,
                                           dispersion=sigma**2)

        result = conv.summary(target_spec,
                              ndraw=ndraw,
                              burnin=burnin, 
                              compute_intervals=False)

        MLE_result, observed_info_mean, _ = conv.selective_MLE(target_spec)

        final_estimator = np.asarray(MLE_result['MLE'])
        pval = np.asarray(result['pvalue'])
        
        if target == 'selected':
            true_target = np.linalg.pinv(X[:,nonzero]).dot(X.dot(beta))
        else:
            true_target = beta[nonzero]

        MLE_pivot = ndist.cdf((final_estimator - true_target) / 
                             np.sqrt(np.diag(observed_info_mean)))
        MLE_pivot = 2 * np.minimum(MLE_pivot, 1 - MLE_pivot)
        
        if MLE:
            return MLE_pivot[true_target == 0], MLE_pivot[true_target != 0]
        else:
            return pval[true_target == 0], pval[true_target != 0]
    else:
        return [], []

def test_all_targets(n=100, p=20, signal_fac=1.5, s=5, sigma=3, rho=0.4):
    for target in ['full', 'selected', 'debiased']:
        test_split_lasso(n=n, 
                         p=p, 
                         signal_fac=signal_fac, 
                         s=s, 
                         sigma=sigma, 
                         rho=rho, 
                         target=target)

