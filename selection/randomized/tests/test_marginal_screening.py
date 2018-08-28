import numpy as np
from scipy.stats import norm as ndist

from ...tests.instance import gaussian_instance
from ..screening import marginal_screening
from ..lasso import lasso

def test_marginal(n=500, 
                  p=50, 
                  s=5, 
                  sigma=3, 
                  rho=0.4, 
                  randomizer_scale=0.5,
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

        q = 0.1
        marginal_select = marginal_screening.type1(score,
                                                   W * sigma**2,
                                                   q,
                                                   randomizer_scale * sigma)

        boundary = marginal_select.fit()
        nonzero = boundary != 0

        if nonzero.sum() > 0:


            if marginal:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = marginal_select.marginal_targets(nonzero)
            else:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = marginal_select.multivariate_targets(nonzero, dispersion=sigma**2)

            if use_MLE:
                estimate, _, _, pval, intervals, _ = marginal_select.selective_MLE(observed_target,
                                                                                   cov_target,
                                                                                   crosscov_target_score)
            # run summary
            else:
                _, pval, intervals = marginal_select.summary(observed_target, 
                                                             cov_target, 
                                                             crosscov_target_score, 
                                                             alternatives,
                                                             compute_intervals=True)

            print(pval)
            if marginal:
                beta_target = true_mean[nonzero]
            else:
                beta_target = beta[nonzero]

            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_simple(n=100,
                p=20,
                s=3,
                use_MLE=False):

    while True:
        Z = np.random.standard_normal(p)
        beta = (2 * np.random.binomial(1, 0.5, size=(p,)) - 1) * 5
        beta[s:] = 0
        np.random.shuffle(beta)

        true_mean = beta
        score = Z + true_mean

        idx = np.arange(p)

        q = 0.1

        marginal_select = marginal_screening.type1(score,
                                                   np.identity(p),
                                                   q,
                                                   1.)

        boundary = marginal_select.fit()
        nonzero = boundary != 0

        # compare to LASSO
        # should have same affine constraints

        perturb = marginal_select._initial_omega # randomization used

        randomized_lasso = lasso.gaussian(np.identity(p),
                                          score,
                                          marginal_select.threshold,
                                          randomizer_scale=1.,
                                          ridge_term=0.)
        
        randomized_lasso.fit(perturb = perturb)

        np.testing.assert_allclose(randomized_lasso.sampler.affine_con.mean, 
                                   marginal_select.sampler.affine_con.mean)

        np.testing.assert_allclose(randomized_lasso.sampler.affine_con.covariance, 
                                   marginal_select.sampler.affine_con.covariance)

        np.testing.assert_allclose(randomized_lasso.sampler.affine_con.linear_part, 
                                   marginal_select.sampler.affine_con.linear_part)

        np.testing.assert_allclose(randomized_lasso.sampler.affine_con.offset, 
                                   marginal_select.sampler.affine_con.offset)

        if nonzero.sum() > 0:

            (observed_target, 
             cov_target, 
             crosscov_target_score, 
             alternatives) = marginal_select.marginal_targets(nonzero)

            if use_MLE:
                estimate, _, _, pval, intervals, _ = marginal_select.selective_MLE(observed_target,
                                                                                   cov_target,
                                                                                   crosscov_target_score)
            # run summary
            else:
                _, pval, intervals = marginal_select.summary(observed_target, 
                                                             cov_target, 
                                                             crosscov_target_score, 
                                                             alternatives,
                                                             compute_intervals=True)

            print(pval)
            beta_target = cov_target.dot(true_mean[nonzero])
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_both():
    test_marginal(marginal=True)
    test_marginal(marginal=False)

def main(nsim=1000, test_fn=test_marginal, use_MLE=False):

    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    U = np.linspace(0, 1, 101)
    P0, PA, cover, length_int = [], [], [], []
    for i in range(nsim):
        p0, pA, cover_, intervals = test_fn(use_MLE=use_MLE)

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')

        if i % 50 == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P0)(U), 'b', label='null')
            plt.plot(U, sm.distributions.ECDF(PA)(U), 'r', label='alt')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.savefig('marginal_screening_pvals.pdf')

