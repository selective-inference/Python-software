import numpy as np
from selection.randomized.marginal_screening import BH, marginal_screening
from selection.tests.instance import gaussian_instance

def test_BH(n=500, 
            p=100, 
            signal_fac=1.6, 
            s=5, 
            sigma=3, 
            rho=0.4, 
            randomizer_scale=0.25,
            run_summary=True):

    while True:
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

        q = 0.1
        BH_select = BH(-X.T.dot(Y),
                       sigma**2 * X.T.dot(X),
                       randomizer_scale * sigma,
                       q)

        boundary = BH_select.fit()
        nonzero = boundary != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            observed_target, cov_target, crosscov_target_score, alternatives = BH_select.form_targets(nonzero)
            estimate, _, _, pval, intervals, _ = BH_select.selective_MLE(observed_target,
                                                                         cov_target,
                                                                         crosscov_target_score)

            # run summary

            if run_summary:
                BH_select.summary(observed_target, 
                                  cov_target, 
                                  crosscov_target_score, 
                                  alternatives,
                                  compute_intervals=True)

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_marginal(n=500, 
                  p=100, 
                  signal_fac=1.6, 
                  s=5, 
                  sigma=3, 
                  rho=0.4, 
                  randomizer_scale=0.25,
                  run_summary=True):

    while True:
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

        q = 0.1
        marginal_select = marginal_screening(-X.T.dot(Y),
                                              sigma**2 * X.T.dot(X),
                                              randomizer_scale * sigma,
                                              q)

        boundary = marginal_select.fit()
        nonzero = boundary != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            observed_target, cov_target, crosscov_target_score, alternatives = marginal_select.form_targets(nonzero)
            estimate, _, _, pval, intervals, _ = marginal_select.selective_MLE(observed_target,
                                                                         cov_target,
                                                                         crosscov_target_score)

            # run summary

            if run_summary:
                marginal_select.summary(observed_target, 
                                        cov_target, 
                                        crosscov_target_score, 
                                        alternatives,
                                        compute_intervals=True)

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def main(nsim=5000, BH=True):

    P0, PA, cover, length_int = [], [], [], []
    for i in range(nsim):
        if BH:
            p0, pA, cover_, intervals = test_BH(run_summary=False)
        else:
            p0, pA, cover_, intervals = test_marginal(run_summary=False)

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')


