import numpy as np

from selection.lasso import covtest

def test_covtest():

    n, p = 30, 50
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,None]
    X /= X.std(0)[None,:]
    Y = np.random.standard_normal(n) * 1.5 

    con, pval = covtest(X, Y, sigma=1.5)

    return pval
