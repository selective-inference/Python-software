import numpy as np
import nose.tools as nt

from ..lasso import lasso, full_targets, selected_targets, debiased_targets
from ...tests.instance import gaussian_instance

def test_full_targets(n=200, 
                      p=1000, 
                      signal_fac=0.5, 
                      s=5, sigma=3, 
                      rho=0.4, 
                      randomizer_scale=0.5,
                      full_dispersion=False):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    while True:
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
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

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

        if nonzero.sum() > 0:
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = None

            if n>p:
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = full_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)
            else:
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = debiased_targets(conv.loglike,
                                                  conv._W,
                                                  nonzero,
                                                  penalty=conv.penalty,
                                                  dispersion=dispersion)

            estimate, _, _, pval, intervals, _ = conv.selective_MLE(observed_target,
                                                                    cov_target,
                                                                    cov_target_score)

            print("estimate, intervals", estimate, intervals)

            coverage = (beta[nonzero] > intervals[:, 0]) * (beta[nonzero] < intervals[:, 1])
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals


def test_selected_targets(n=2000, 
                          p=200, 
                          signal_fac=1., 
                          s=5, 
                          sigma=3, 
                          rho=0.4, 
                          randomizer_scale=1,
                          full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
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
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        if nonzero.sum() > 0:
            dispersion = None
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero, 
                                              dispersion=dispersion)

            estimate, _, _, pval, intervals, _ = conv.selective_MLE(observed_target,
                                                                    cov_target,
                                                                    cov_target_score)

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals


def main(nsim=500, full=False):
    P0, PA, cover, length_int = [], [], [], []
    from statsmodels.distributions import ECDF

    n, p, s = 500, 100, 10

    for i in range(nsim):
        if full:
            if n > p:
                full_dispersion = True
            else:
                full_dispersion = False
            p0, pA, cover_, intervals = test_full_targets(n=n, p=p, s=s, full_dispersion=full_dispersion)
            avg_length = intervals[:, 1] - intervals[:, 0]
        else:
            full_dispersion = True
            p0, pA, cover_, intervals = test_selected_targets(n=n, p=p, s=s,
                                                              full_dispersion=full_dispersion)
            avg_length = intervals[:, 1] - intervals[:, 0]

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(
            np.array(PA) < 0.1, np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.1), np.mean(np.array(PA) < 0.1), np.mean(cover),
            np.mean(avg_length), 'null pvalue + power + length')

