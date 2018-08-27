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

        X = gaussian_instance(n=n,
                              p=p,
                              equicorrelated=False,
                              rho=rho)[0]
        W = rho**(np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
        sqrtW = np.linalg.cholesky(W)
        sigma = 0.15
        Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
        beta = (2 * np.random.binomial(1, 0.5, size=(p,)) - 1) * 5 * sigma
        beta[s:] = 0
        np.random.shuffle(beta)

        true_mean = W.dot(beta)
        score = Z + true_mean
        idx = np.arange(p)

        n, p = X.shape

        q = 0.1
        BH_select = stepup.BH(score,
                              W * sigma**2,
                              randomizer_scale * sigma,
                              q=q)

        boundary = BH_select.fit()

        if boundary is not None:
            nonzero = boundary != 0
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

def main(nsim=500, use_MLE=False):

    P0, PA, cover, length_int = [], [], [], []
    for i in range(nsim):
        p0, pA, cover_, intervals = test_BH(use_MLE=use_MLE)

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')


