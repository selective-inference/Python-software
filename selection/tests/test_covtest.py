import numpy as np

from selection.lasso import covtest

def test_covtest():

    n, p = 30, 50
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n) + X[:,1] * 0.2

    con, pval = covtest(X, Y)

    return pval
