import numpy as np
from selection.sqrt_lasso import sqrt_lasso, choose_lambda

def test_class(n=100, p=20, frac=0.5):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    lam_theor = choose_lambda(X, quantile=0.25)
    L = sqrt_lasso(y,X,lam_theor)
    L.fit(tol=1.e-7)
    print L.active_pvalues

    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L.y),
        L.constraints.offset)

    #I = L.intervals
    P = L.active_pvalues

    return P #, I

if __name__ == "__main__":
    test_class()
