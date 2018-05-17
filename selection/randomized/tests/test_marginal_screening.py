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

def test_marginal_screening(n=500, 
                            p=100, 
                            signal_fac=1.6, 
                            s=5, 
                            sigma=3, 
                            rho=0.4, 
                            randomizer_scale=1.):

    if True:
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

        #idx = np.arange(p)
        #sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        #print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))
        
        X0 = X - X.mean(0)[None,:]
        Y0 = Y - Y.mean()
        beta_ols = np.linalg.pinv(X0).dot(Y0)
        dispersion = ((Y0 - X0.dot(beta_ols))**2).sum() / (n-p-1)

        n, p = X.shape

        q = 0.1
        #marginal_select = marginal_screening(-X.T.dot(Y),
        #                                      sigma**2 * X.T.dot(X),
        #                                      randomizer_scale * sigma,
        #                                      q)
        
        marginal_select = marginal_screening(-X.T.dot(Y),
                                              dispersion * X.T.dot(X),
                                              randomizer_scale * np.sqrt(dispersion),
                                              q)

        boundary = marginal_select.fit()
        nonzero = boundary != 0
        #print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            #observed_target, cov_target, crosscov_target_score, alternatives = marginal_select.form_targets(nonzero, sigma**2)
            observed_target, cov_target, crosscov_target_score, alternatives = marginal_select.form_targets(nonzero, dispersion)
            
            # run summary
            _, pval, intervals = marginal_select.summary(observed_target, 
                                                         cov_target, 
                                                         crosscov_target_score, 
                                                         alternatives,
                                                         compute_intervals=True,
                                                         ndraw = 2000,
                                                         burnin = 1000)

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals, dispersion

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


