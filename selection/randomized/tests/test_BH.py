import numpy as np
import numpy.testing.decorators as dec

from scipy.stats import norm as ndist

from ...tests.instance import gaussian_instance
from ...tests.decorators import rpy_test_safe

from ..screening import stepup
from ..screening import stepup, stepup_selection
from ..randomization import randomization

def BHfilter(pval, q=0.2):
    pval = np.asarray(pval)
    pval_sort = np.sort(pval)
    comparison = q * np.arange(1, pval.shape[0] + 1.) / pval.shape[0]
    passing = pval_sort < comparison
    if passing.sum():
        thresh = comparison[np.nonzero(passing)[0].max()]
        return np.nonzero(pval <= thresh)[0]
    return []

@rpy_test_safe()
def test_BH_procedure():

    def BH_cutoff():
        Z = np.random.standard_normal(100)

        BH = stepup.BH(Z,
                       np.identity(100),
                       1.)

        cutoff = BH.stepup_Z / np.sqrt(2)
        return cutoff
    
    BH_cutoffs = BH_cutoff()

    for _ in range(50):
        Z = np.random.standard_normal(100)
        Z[:20] += 3

        np.testing.assert_allclose(sorted(BHfilter(2 * ndist.sf(np.fabs(Z)), q=0.2)),
                                   sorted(stepup_selection(Z, BH_cutoffs)[1]))

@dec.skipif(True, "independent estimator test not working")
def test_independent_estimator(n=100, n1=50, q=0.2, signal=3, p=100):

    Z = np.random.standard_normal((n, p))
    Z[:, :10] += signal / np.sqrt(n)
    Z1 = Z[:n1]
    
    Zbar = np.mean(Z, 0)
    Zbar1 = np.mean(Z1, 0)
    perturb = Zbar1 - Zbar
    
    frac = n1 * 1. / n
    BH_select = stepup.BH(Zbar, np.identity(p) / n, np.sqrt((1 - frac) / (n * frac)), q=q)
    selected = BH_select.fit(perturb=perturb)
    
    observed_target = Zbar[selected]
    cov_target = np.identity(selected.sum()) / n
    cross_cov = -np.identity(p)[selected] / n

    observed_target1, cov_target1, cross_cov1, _ = BH_select.marginal_targets(selected)

    assert(np.linalg.norm(observed_target - observed_target1) / np.linalg.norm(observed_target) < 1.e-7)
    assert(np.linalg.norm(cov_target - cov_target1) / np.linalg.norm(cov_target) < 1.e-7)
    assert(np.linalg.norm(cross_cov - cross_cov1) / np.linalg.norm(cross_cov) < 1.e-7)

    (final_estimator, 
     _, 
     Z_scores, 
     pvalues, 
     intervals, 
     ind_unbiased_estimator) = BH_select.selective_MLE(observed_target, cov_target, cross_cov)

    Zbar2 = Z[n1:].mean(0)[selected]

    assert(np.linalg.norm(ind_unbiased_estimator - Zbar2) / np.linalg.norm(Zbar2) < 1.e-6)
    np.testing.assert_allclose(sorted(np.nonzero(selected)[0]), 
                               sorted(BHfilter(2 * ndist.sf(np.fabs(np.sqrt(n1) * Zbar1)))))


def test_BH(n=500, 
            p=500, 
            s=50, 
            sigma=3, 
            rho=0.65, 
            randomizer_scale=1.,
            use_MLE=True,
            marginal=False,
            level=0.9):

    while True:

        W = rho**(np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
        sqrtW = np.linalg.cholesky(W)
        sigma = 0.5
        Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
        beta = (2 * np.random.binomial(1, 0.5, size=(p,)) - 1) * np.linspace(4, 5, p) * sigma
        np.random.shuffle(beta)
        beta[s:] = 0
        np.random.shuffle(beta)
        print(beta, 'beta')

        true_mean = W.dot(beta)
        score = Z + true_mean

        q = 0.1
        BH_select = stepup.BH(score,
                              W * sigma**2,
                              randomizer_scale * sigma,
                              q=q)

        nonzero = BH_select.fit()

        if nonzero is not None:

            if marginal:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = BH_select.marginal_targets(nonzero)
            else:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = BH_select.full_targets(nonzero, dispersion=sigma**2)
               
            if marginal:
                beta_target = true_mean[nonzero]
            else:
                beta_target = beta[nonzero]

            if use_MLE:
                print('huh')
                estimate, info, _, pval, intervals, _ = BH_select.selective_MLE(observed_target,
                                                                                cov_target,
                                                                                crosscov_target_score,
                                                                                level=level)
                pivots = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(info)))
                pivots = 2 * np.minimum(pivots, 1 - pivots)
                # run summary
            else:
                pivots, pval, intervals = BH_select.summary(observed_target, 
                                                            cov_target, 
                                                            crosscov_target_score, 
                                                            alternatives,
                                                            compute_intervals=True,
                                                            level=level,
                                                            ndraw=20000,
                                                            burnin=2000,
                                                            parameter=beta_target)
            print(pval)
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pivots[beta_target == 0], pivots[beta_target != 0], coverage, intervals, pivots
        else:
            return [], [], [], [], []

def test_both():
    test_BH(marginal=True)
    test_BH(marginal=False)

def main(nsim=500, use_MLE=True, marginal=False):

    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    U = np.linspace(0, 1, 101)
    P0, PA, cover, length_int = [], [], [], []
    Ps = []
    for i in range(nsim):
        p0, pA, cover_, intervals, pivots = test_BH(use_MLE=use_MLE, marginal=marginal)
        Ps.extend(pivots)
        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')

        period = 10
        if use_MLE:
            period = 50
        if i % period == 0 and i > 0:
            plt.clf()
            if len(P0) > 0:
                plt.plot(U, sm.distributions.ECDF(P0)(U), 'b', label='null')
            plt.plot(U, sm.distributions.ECDF(PA)(U), 'r', label='alt')
            plt.plot(U, sm.distributions.ECDF(Ps)(U), 'tab:orange', label='pivot')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.legend()
            plt.savefig('BH_pvals.pdf')


