import numpy as np
import nose.tools as nt
import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
#rpy.r('library(selectiveInference)')

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import lasso
from selection.tests.instance import gaussian_instance
import matplotlib.pyplot as plt

def test_full_targets(n=2000, p=200, signal_fac=0.5, s=5, sigma=3, rho=0.4, randomizer_scale=0.25, full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s,
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    idx = np.arange(p)
    sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
    print("snr", beta.T.dot(sigmaX).dot(beta)/((sigma**2.)* n))

    n, p = X.shape

    sigma_ = np.std(Y)
    W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer_scale=randomizer_scale * sigma_)
    
    signs = conv.fit()
    nonzero = signs != 0
    print("dimensions", n, p, nonzero.sum())

    dispersion = None
    if full_dispersion:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y)))**2 / (n - p)

    estimate, _, _, pval, intervals, _ = conv.selective_MLE(target="full", dispersion=dispersion)

    coverage = (beta[nonzero] > intervals[:,0]) * (beta[nonzero] < intervals[:,1])
    return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_selected_targets(n=2000, p=200, signal_fac=1.5, s=5, sigma=3, rho=0.4, randomizer_scale=1, full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))
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
    W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer_scale=randomizer_scale * sigma_)
    
    signs = conv.fit()
    nonzero = signs != 0

    dispersion = None
    if full_dispersion:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y)))**2 / (n - p)

    estimate, _, _, pval, intervals, _ = conv.selective_MLE(target="selected", dispersion=dispersion)

    beta_target = np.linalg.pinv(X[:,nonzero]).dot(X.dot(beta))

    coverage = (beta_target > intervals[:,0]) * (beta_target < intervals[:,1])
    return pval[beta_target == 0], pval[beta_target != 0], coverage

def main(nsim=500, full=True):

    P0, PA, cover, length_int = [], [], [], []
    from statsmodels.distributions import ECDF

    n, p, s = 200, 1000, 10

    for i in range(nsim):
        if full:
            if n>p:
                full_dispersion = True
            else:
                full_dispersion = False
            p0, pA, cover_, intervals = test_full_targets(n=n, p=p, s=s, full_dispersion=full_dispersion)
            avg_length = intervals[:,1]-intervals[:,0]
        else:
            full_dispersion = True
            p0, pA, cover_ = test_selected_targets(n=n, p=p, s=s, full_dispersion=full_dispersion)

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.1), np.mean(np.array(PA) < 0.1), np.mean(cover),
              np.mean(avg_length), 'null pvalue + power + length')
    
        if i % 3 == 0 and i > 0:
            U = np.linspace(0, 1, 101)
            plt.clf()
            if len(P0) > 0:
                plt.plot(U, ECDF(P0)(U))
            if len(PA) > 0:
                plt.plot(U, ECDF(PA)(U), 'r')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.savefig("/Users/snigdhapanigrahi/Desktop/plot.pdf")
    plt.show()

main()

