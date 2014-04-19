import numpy as np
from selection.lasso import lasso # , _howlong

def test_class(n=100, p=20, frac=0.9):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    L = lasso(y,X,frac=frac)
    L.fit(tol=1.e-7)
    L.form_constraints()
    C = L.constraints

    np.testing.assert_array_less(np.dot(L.constraints.inequality, L.y), L.constraints.inequality_offset)

    I = L.intervals
    P = L.active_pvalues

    return L.centered_test, L.basic_test, L, C, I, P

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
