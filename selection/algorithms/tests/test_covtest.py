import numpy as np
import itertools
import numpy.testing.decorators as dec
from matplotlib import pyplot as plt

from selection.algorithms.lasso import instance, lasso
from selection.algorithms.covtest import covtest, selected_covtest
from selection.constraints.affine import gibbs_test

def test_covtest():

    n, p = 30, 50
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,None]
    X /= X.std(0)[None,:]
    Y = np.random.standard_normal(n) * 1.5 

    for exact, covariance in itertools.product([True, False],
                                               [None, np.identity(n)]):
        con, pval, idx, sign = covtest(X, Y, sigma=1.5, exact=exact,
                                       covariance=covariance)
    for covariance in [None, np.identity(n)]:
        con, pval, idx, sign = selected_covtest(X, Y, sigma=1.5,
                                                covariance=covariance)

    return pval

@dec.slow
def test_tilting(nsim=100):

    P = []
    for _ in range(nsim):
        X, Y, beta, active, sigma = instance()
        Y0 = np.random.standard_normal(X.shape[0]) * sigma
        cone, pvalue, idx, sign = selected_covtest(X, Y0, sigma=sigma)
        p1 = gibbs_test(cone, Y0, X[:,idx] * sign,
                        ndraw=25000,
                        burnin=1000,
                        alternative='greater',
                        sigma_known=True)[0]

        p2 = gibbs_test(cone, Y0, X[:,idx] * sign,
                        ndraw=25000,
                        burnin=1000,
                        alternative='greater',
                        sigma_known=True,
                        do_tilt=False)[0]
        P.append((p1, p2))
        Pa = np.array(P)

        # p1 and p2 should be very close, so have high correlation
        print np.corrcoef(Pa.T)[0,1]

        # they should also look uniform -- mean should be about 0.5, sd about 0.29

        print np.mean(Pa, 0), np.std(Pa, 0)
    plt.figure()
    plt.scatter(Pa[:,0], Pa[:,1])
