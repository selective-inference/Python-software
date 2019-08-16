from __future__ import division, print_function

import numpy as np
import nose.tools as nt

from scipy.stats import norm as ndist

import regreg.api as rr

from ..lasso import (split_lasso, 
                     selected_targets, 
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
            (observed_target, 
             cov_target, 
             cov_target_score, 
             alternatives) = full_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          dispersion=sigma**2)
        elif target == 'selected':
            (observed_target, 
             cov_target, 
             cov_target_score, 
             alternatives) = selected_targets(conv.loglike, 
                                              conv._W, 
                                              nonzero) #,
                                              #dispersion=sigma**2)

        elif target == 'debiased':
            (observed_target, 
             cov_target, 
             cov_target_score, 
             alternatives) = debiased_targets(conv.loglike, 
                                              conv._W, 
                                              nonzero,
                                              penalty=conv.penalty,
                                              dispersion=sigma**2)

        _, pval, intervals = conv.summary(observed_target, 
                                          cov_target, 
                                          cov_target_score, 
                                          alternatives,
                                          ndraw=ndraw,
                                          burnin=burnin, 
                                          compute_intervals=False)

        final_estimator, observed_info_mean = conv.selective_MLE(
                                                 observed_target,
                                                 cov_target,
                                                 cov_target_score)[:2]
        
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

def main(nsim=500, n=100, p=200, target='selected', sigma=3, s=3):

    import matplotlib.pyplot as plt
    P0, PA = [], []
    from statsmodels.distributions import ECDF

    for i in range(nsim):
        p0, pA = test_split_lasso(n=n, p=p, target=target, sigma=sigma, s=s)
        print(len(p0), len(pA))
        if not (len(pA) < s and target=='selected'):
            P0.extend(p0)
            PA.extend(pA)

        P0_clean = np.array(P0)
        
        P0_clean = P0_clean[P0_clean > 1.e-5] # 
        print(np.mean(P0_clean), np.std(P0_clean), np.mean(np.array(PA) < 0.05), np.sum(np.array(PA) < 0.05) / (i+1), np.mean(np.array(P0) < 0.05), np.mean(P0_clean < 0.05), np.mean(np.array(P0) < 1e-5), 'null pvalue + power + failure')
    
        if i % 3 == 0 and i > 0:
            U = np.linspace(0, 1, 101)
            plt.clf()
            if len(P0_clean) > 0:
                plt.plot(U, ECDF(P0_clean)(U))
            if len(PA) > 0:
                plt.plot(U, ECDF(PA)(U), 'r')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.savefig("plot.pdf")
    plt.show()


