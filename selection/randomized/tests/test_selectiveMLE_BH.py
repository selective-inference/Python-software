import numpy as np
from selection.randomized.marginal_screening import BH
from selection.tests.instance import gaussian_instance

def test_selected_targets(n=500, p=100, signal_fac=1.6, s=5, sigma=3, rho=0.4, randomizer_scale=0.25,
                          full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst = gaussian_instance
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

    conv = BH.gaussian(X,
                       Y,
                       sigma = sigma_,
                       randomizer_scale=randomizer_scale * sigma_)

    boundary = conv.fit()
    nonzero = boundary != 0
    print("dimensions", n, p, nonzero.sum())

    dispersion = None
    if full_dispersion:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

    estimate, _, _, pval, intervals, _ = conv.selective_MLE(target="selected", dispersion=dispersion)

    beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
    print("beta_target and intervals", beta_target, intervals)
    coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
    print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
    return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def main(nsim=500):

    P0, PA, cover, length_int = [], [], [], []
    for i in range(nsim):
        p0, pA, cover_, intervals = test_selected_targets()

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')

main()
