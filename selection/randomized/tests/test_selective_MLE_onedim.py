import numpy as np
import nose.tools as nt

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim 
from selection.tests.instance import gaussian_instance
import matplotlib.pyplot as plt

def test_onedim_lasso(n=200, p=1, signal_fac=1.5, s=1, ndraw=5000, burnin=1000, sigma=3, full=True, rho=0.4, randomizer_scale=1):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, highdim.gaussian
    signal = signal_fac * np.sqrt(2 * np.log(p+1.))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    n, p = X.shape

    W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p+1.)) * sigma

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer_scale=randomizer_scale * sigma)
    
    signs = conv.fit()
    nonzero = signs != 0

    estimate, _, _, pv = conv.selective_MLE(target="full")
    print(estimate, 'selective MLE')
    print(beta[nonzero], 'truth')
    print(np.linalg.pinv(X[:,nonzero]).dot(Y), 'relaxed')
    print(pv[beta[nonzero] == 0], pv[beta[nonzero] != 0])

    if full:
        _, pval, intervals = conv.summary(target="full",
                                          ndraw=ndraw,
                                          burnin=burnin, 
                                          compute_intervals=False)
    else:
        _, pval, intervals = conv.summary(target="selected",
                                          ndraw=ndraw,
                                          burnin=burnin, 
                                          compute_intervals=False)

    return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0]


def main(nsim=500):

    P0, PA = [], []
    from statsmodels.distributions import ECDF

    n, p = 500, 200

    for i in range(nsim):
        try:
            p0, pA = test_highdim_lasso(n=n, p=p, full=True)
        except:
            p0, pA = [], []
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(P0), np.std(P0), np.mean(np.array(PA) < 0.05))
    
        if i % 3 == 0 and i > 0:
            U = np.linspace(0, 1, 101)
            plt.clf()
            if len(P0) > 0:
                plt.plot(U, ECDF(P0)(U))
            if len(PA) > 0:
                plt.plot(U, ECDF(PA)(U), 'r')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.savefig("plot.pdf")
    plt.show()

