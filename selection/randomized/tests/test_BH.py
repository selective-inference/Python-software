import numpy as np
from scipy.stats import norm as ndist

from ..stepup import stepup
from ...tests.instance import gaussian_instance

def test_BH(n=500, 
            p=100, 
            signal_fac=1.6, 
            s=5, 
            sigma=3, 
            rho=0.4, 
            randomizer_scale=0.25,
            use_MLE=True):

    while True:
        inst = gaussian_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))
        X, Y, beta, _, sigma, sigmaX = inst(n=n,
                                            p=p,
                                            signal=signal,
                                            s=s,
                                            equicorrelated=False,
                                            rho=rho,
                                            sigma=sigma,
                                            random_signs=True,
                                            scale=True)

        idx = np.arange(p)

        n, p = X.shape

        q = 0.1
        BH_select = stepup.BH(X.T.dot(Y),
                              sigma**2 * sigmaX,
                              randomizer_scale * sigma,
                              q)

        boundary = BH_select.fit()
        nonzero = boundary != 0

        if nonzero.sum() > 0:
            (observed_target, 
             cov_target, 
             crosscov_target_score, 
             alternatives) = BH_select.marginal_targets(nonzero)
            if use_MLE:
                estimate, _, _, pval, intervals, _ = BH_select.selective_MLE(observed_target,
                                                                             cov_target,
                                                                             crosscov_target_score)
                # run summary
            else:
                _, pval, intervals = BH_select.summary(observed_target, 
                                                       cov_target, 
                                                       crosscov_target_score, 
                                                       alternatives,
                                                       compute_intervals=True)

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def main(nsim=5000, use_MLE=False):

    P0, PA, cover, length_int = [], [], [], []
    for i in range(nsim):
        p0, pA, cover_, intervals = test_BH(use_MLE=use_MLE)

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')


