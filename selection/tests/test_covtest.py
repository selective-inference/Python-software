import numpy as np
import itertools
from selection.covtest import covtest

def test_covtest():

    n, p = 30, 50
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,None]
    X /= X.std(0)[None,:]
    Y = np.random.standard_normal(n) * 1.5 

    for exact, covariance in itertools.product([True, False],
                                               [None, np.identity(n)]):
        con, pval, idx, sign = covtest(X, Y, sigma=1.5, exact=exact,
                                       covariance=covariance)

    return pval
