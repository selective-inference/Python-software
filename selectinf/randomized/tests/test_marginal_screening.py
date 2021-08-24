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
                target_spec = marginal_select.marginal_targets(nonzero)
            else:
                target_spec = marginal_select.multivariate_targets(nonzero, dispersion=sigma**2)

            if use_MLE:
                result = marginal_select.selective_MLE(target_spec)[0]
            # run summary
            else:
                result = marginal_select.summary(target_spec,
                                                 compute_intervals=True)

            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
            pval = result['pvalue']
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

            target_spec = marginal_select.marginal_targets(nonzero)

            if use_MLE:
                result = marginal_select.selective_MLE(target_spec)
            # run summary
            else:
                result = marginal_select.summary(target_spec,
                                                 compute_intervals=True)

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
            print(pval)
            beta_target = target_spec.cov_target.dot(true_mean[nonzero])
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_both():
    test_marginal(marginal=True)
    test_marginal(marginal=False)
