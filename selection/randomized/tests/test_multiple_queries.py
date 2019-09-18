from __future__ import division, print_function

import numpy as np
import nose.tools as nt

import regreg.api as rr

from ..lasso import lasso, selected_targets, full_targets, debiased_targets
from ..screening import marginal_screening
from ..query import multiple_queries
from ...tests.instance import gaussian_instance
from ...algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso

# the test here is marginal_screening + lasso
def test_multiple_queries(n=500, p=100, signal_fac=1.5, s=5, sigma=3, rho=0.4, randomizer_scale=1, ndraw=5000, burnin=1000):

    inst, const1, const2 = gaussian_instance, marginal_screening, lasso.gaussian
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

    q = 0.1
    conv1 = const1.type1(-X.T.dot(Y),
                          sigma**2 * X.T.dot(X),
                          q,
                          randomizer_scale * sigma)

    boundary1 = conv1.fit()
    nonzero1 = boundary1 != 0

    sigma_ = np.std(Y)
    W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma_

    conv2 = const2(X, 
                   Y, 
                   W, 
                   randomizer_scale=randomizer_scale * sigma_)
    
    signs2 = conv2.fit()
    nonzero2 = signs2 != 0

    nonzero = nonzero1 * nonzero2

    if nonzero.sum() == 0:
      return [], []

    observed_target1, cov_target1, cov_target_score1, alternatives1 = conv1.multivariate_targets(nonzero, sigma**2)

    (observed_target2, 
     cov_target2, 
     cov_target_score2, 
     alternatives2) = selected_targets(conv2.loglike, 
                                       conv2._W, 
                                       nonzero)

    mq = multiple_queries([conv1, conv2])

    _, pval, intervals = mq.summary(observed_target1, 
                                   [(cov_target1, cov_target_score1), (cov_target2, cov_target_score2)],
                                    compute_intervals=True)
        
    return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0]


def main(nsim=500, n=500, p=100, sigma=3):

    P0, PA = [], []
    from statsmodels.distributions import ECDF
    import matplotlib.pyplot as plt

    for i in range(nsim):
        if True:
          p0, pA = test_multiple_queries(n=n, p=p, sigma=sigma)
        else: 
          p0, pA = [], []
        P0.extend(p0)
        PA.extend(pA)

        P0_clean = np.array(P0)
        
        P0_clean = P0_clean[P0_clean > 1.e-5] # 
        print(np.mean(P0_clean), np.std(P0_clean), np.mean(np.array(PA) < 0.05), np.mean(np.array(P0) < 0.05), np.mean(P0_clean < 0.05), np.mean(np.array(P0) < 1e-5))
    
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

