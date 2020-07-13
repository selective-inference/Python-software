import numpy as np
from scipy.stats import norm as ndist

from ..screening import topK
from ..randomization import randomization
from ...tests.instance import gaussian_instance

def test_topK(n=500, 
              p=50, 
              s=5, 
              sigma=3, 
              rho=0.4, 
              randomizer_scale=0.50,
              use_MLE=True,
              marginal=False):

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

        K = 5
        randomizer = randomization.isotropic_gaussian(p, randomizer_scale * sigma)
        topK_select = topK(score,
                           W * sigma**2,
                           randomizer,
                           K)

        boundary = topK_select.fit()
        nonzero = boundary != 0

        if nonzero.sum() > 0:

            if marginal:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = topK_select.marginal_targets(nonzero)
            else:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = topK_select.multivariate_targets(nonzero, dispersion=sigma**2)
               
            if use_MLE:
                result = topK_select.selective_MLE(observed_target,
                                                   cov_target,
                                                   crosscov_target_score)[0]
            # run summary
            else:
                result = topK_select.summary(observed_target, 
                                             cov_target, 
                                             crosscov_target_score, 
                                             alternatives,
                                             compute_intervals=True)
            lower = np.asarray(result['lower_confidence'])
            upper = np.asarray(result['upper_confidence'])
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
            print(pval)
            if marginal:
                beta_target = true_mean[nonzero]
            else:
                beta_target = beta[nonzero]
            print("beta_target and intervals", beta_target, lower, upper)
            coverage = (beta_target > lower) * (beta_target < upper)
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_both():
    test_topK(marginal=True)
    test_topK(marginal=False)

def test_bias_topK(n=500,
                   p=50,
                   s=5,
                   sigma=3,
                   rho=0.4,
                   randomizer_scale=0.50,
                   K=5,
                   marginal=False):

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

        randomizer = randomization.isotropic_gaussian(p, randomizer_scale * sigma)
        topK_select = topK(score,
                           W * sigma**2,
                           randomizer,
                           K)

        boundary = topK_select.fit()
        nonzero = boundary != 0

        if nonzero.sum() > 0:

            if marginal:
                beta_target = true_mean[nonzero]
                (observed_target,
                 cov_target,
                 crosscov_target_score,
                 alternatives) = topK_select.marginal_targets(nonzero)
            else:
                beta_target = beta[nonzero]
                (observed_target,
                 cov_target,
                 crosscov_target_score,
                 alternatives) = topK_select.multivariate_targets(nonzero, dispersion=sigma**2)

            result = topK_select.selective_MLE(observed_target,
                                               cov_target,
                                               crosscov_target_score)[0]

            bias_mle = np.asarray(result['MLE'])-beta_target
            bias_indest = np.asarray(result['unbiased'])-beta_target
            print("check ", np.asarray(result['MLE']), np.asarray(result['unbiased']))

            return bias_mle, bias_indest


# def main(nsim=5000, use_MLE=False):
#
#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm
#     U = np.linspace(0, 1, 101)
#
#     P0, PA, cover, length_int = [], [], [], []
#     for i in range(nsim):
#         p0, pA, cover_, intervals = test_topK(use_MLE=use_MLE)
#
#         cover.extend(cover_)
#         P0.extend(p0)
#         PA.extend(pA)
#         print(np.mean(cover),'coverage so far')
#
#         period = 10
#         if use_MLE:
#             period = 50
#         if i % period == 0 and i > 0:
#             plt.clf()
#             plt.plot(U, sm.distributions.ECDF(P0)(U), 'b', label='null')
#             plt.plot(U, sm.distributions.ECDF(PA)(U), 'r', label='alt')
#             plt.plot([0, 1], [0, 1], 'k--')
#             plt.legend()
#             plt.savefig('topK_pvals.pdf')


def main(nsim=500):
    _bias_mle = []
    _bias_indest = []

    for i in range(nsim):
        bias_mle, bias_indest = test_bias_topK()
        _bias_mle.extend(bias_mle)
        _bias_indest.extend(bias_indest)

        print(np.mean(_bias_mle), np.mean(_bias_indest), 'bias so far: mle and independent estimate ')


if __name__ == "__main__":
    main(nsim=500)
