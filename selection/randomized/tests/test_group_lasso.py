from __future__ import division, print_function

import numpy as np
import nose.tools as nt

import regreg.api as rr

from ..group_lasso import (group_lasso,
                           selected_targets, 
                           full_targets, 
                           debiased_targets)
from ...tests.instance import gaussian_instance
from ...tests.flags import SET_SEED
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from ...algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
from ..randomization import randomization
from ...tests.decorators import rpy_test_safe

@set_seed_iftrue(SET_SEED)
def test_group_lasso(n=400, 
                     p=100, 
                     signal_fac=3, 
                     s=5, 
                     sigma=3, 
                     target='full',
                     rho=0.4, 
                     randomizer_scale=.75,
                     ndraw=100000):
    """
    Test group lasso
    """

    inst, const = gaussian_instance, group_lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))

    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    orthogonal = True
    if orthogonal:
        X = np.linalg.svd(X, full_matrices=False)[0] 
        Y = X.dot(beta) + sigma * np.random.standard_normal(n)

    n, p = X.shape

    sigma_ = np.std(Y)

    groups = np.floor(np.arange(p)/2).astype(np.int)
    weights = dict([(i, sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])
    conv = const(X, 
                 Y, 
                 groups,
                 weights, 
                 randomizer_scale=randomizer_scale * sigma_)
    
    signs = conv.fit()
    nonzero = conv.selection_variable['directions'].keys()

    if target == 'full':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = full_targets(conv.loglike, 
                                      conv._W, 
                                      nonzero,
                                      conv.penalty)
    elif target == 'selected':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = selected_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          conv.penalty)
    elif target == 'debiased':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = debiased_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          conv.penalty)

    _, pval, intervals = conv.summary(observed_target, 
                                      group_assignments,
                                      cov_target, 
                                      cov_target_score, 
                                      alternatives,
                                      ndraw=ndraw,
                                      compute_intervals=False)
        
    which = np.zeros(p, np.bool)
    for group in conv.selection_variable['directions'].keys():
        which_group = conv.penalty.groups == group
        which += which_group
    return pval[beta[which] == 0], pval[beta[which] != 0]

@set_seed_iftrue(SET_SEED)
def test_lasso(n=400, 
               p=200, 
               signal_fac=1.5, 
               s=5, 
               sigma=3, 
               target='full',
               rho=0.4, 
               ndraw=10000):
    """
    Test group lasso with groups of size 1, ie lasso
    """

    inst, const = gaussian_instance, group_lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)

    groups = np.arange(p)
    weights = dict([(i, sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])
    conv = const(X, 
                 Y, 
                 groups,
                 weights)
    
    signs = conv.fit()
    nonzero = conv.selection_variable['directions'].keys()

    if target == 'full':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = full_targets(conv.loglike, 
                                      conv._W, 
                                      nonzero,
                                      conv.penalty)
    elif target == 'selected':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = selected_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          conv.penalty)
    elif target == 'debiased':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = debiased_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          conv.penalty)

    _, pval, intervals = conv.summary(observed_target, 
                                      group_assignments,
                                      cov_target, 
                                      cov_target_score, 
                                      alternatives,
                                      ndraw=ndraw,
                                      compute_intervals=False)
        
    which = np.zeros(p, np.bool)
    for group in conv.selection_variable['directions'].keys():
        which_group = conv.penalty.groups == group
        which += which_group
    return pval[beta[which] == 0], pval[beta[which] != 0]

@set_seed_iftrue(SET_SEED)
def test_mixed(n=400, 
               p=200, 
               signal_fac=1.5, 
               s=5, 
               sigma=3, 
               target='full',
               rho=0.4, 
               ndraw=10000):
    """
    Test group lasso with a mix of groups of size 1, and larger
    """

    inst, const = gaussian_instance, group_lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)

    groups = np.arange(p)
    groups[-5:] = -1
    groups[-8:-5] = -2
    Y += X[:,-8:].dot(np.ones(8)) * 5 # so we select the last two groups

    weights = dict([(i, sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])
    conv = const(X, 
                 Y, 
                 groups,
                 weights)
    
    signs = conv.fit()
    nonzero = conv.selection_variable['directions'].keys()

    if target == 'full':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = full_targets(conv.loglike, 
                                      conv._W, 
                                      nonzero,
                                      conv.penalty)
    elif target == 'selected':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = selected_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          conv.penalty)
    elif target == 'debiased':
        (observed_target, 
         group_assignments,
         cov_target, 
         cov_target_score, 
         alternatives) = debiased_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          conv.penalty)

    _, pval, intervals = conv.summary(observed_target, 
                                      group_assignments,
                                      cov_target, 
                                      cov_target_score, 
                                      alternatives,
                                      ndraw=ndraw,
                                      compute_intervals=False)
        
    which = np.zeros(p, np.bool)
    for group in conv.selection_variable['directions'].keys():
        which_group = conv.penalty.groups == group
        which += which_group
    return pval[beta[which] == 0], pval[beta[which] != 0]

@set_seed_iftrue(SET_SEED)
def test_all_targets(n=100, p=20, signal_fac=1.5, s=5, sigma=3, rho=0.4):
    for target in ['full', 'selected', 'debiased']:
        test_group_lasso(n=n, 
                         p=p, 
                         signal_fac=signal_fac, 
                         s=s, 
                         sigma=sigma, 
                         rho=rho, 
                         target=target)

def main(nsim=500, n=200, p=50, target='full', sigma=3):

    import matplotlib.pyplot as plt
    P0, PA = [], []
    from statsmodels.distributions import ECDF

    for i in range(nsim):
        try:
            p0, pA = test_group_lasso(n=n, p=p, target=target, sigma=sigma)
        except:
            pass
        print(len(p0), len(pA))
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


