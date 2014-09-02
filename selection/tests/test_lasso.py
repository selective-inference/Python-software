import numpy as np
from selection.lasso import lasso # , _howlong

def test_class(n=100, p=20, frac=0.5):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    lam_theor = frac * np.fabs(np.dot(X.T, np.random.standard_normal((n,10000)))).max(0).mean()
    L = lasso(y,X,lam=lam_theor)
    L.fit(tol=1.e-7)
    L.form_constraints()
    C = L.constraints

    np.testing.assert_array_less(np.dot(L.constraints.linear_part, L.y), L.constraints.offset)

    I = L.intervals
    P = L.active_pvalues

    return L, C, I, P

# def test_fit_and_test(n=100, p=20, frac=0.9):

#     y = np.random.standard_normal(n)
#     X = np.random.standard_normal((n,p))
#     return _howlong(y, X, frac)

# def test_agreement(n=100, p=20, frac=0.9):

#     y = np.random.standard_normal(n)
#     X = np.random.standard_normal((n,p))
#     P1 = _howlong(y, X, frac)
#     P2 = _howlong(y, X, frac, use_cvx=True)

#     return P1, P2
