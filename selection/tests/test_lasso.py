import numpy as np
from selection.lasso import lasso, fit_and_test

def test_class(n=100, p=20, frac=0.9):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    L = lasso(y,X,frac=frac)
    C = L.constraints
    I = L.intervals
    return L.centered_test, L.basic_test, L, C, I

def test_fit_and_test(n=100, p=20, frac=0.9):

    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    return fit_and_test(y, X, frac)

def test_agreement(n=100, p=20, frac=0.9):

    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    P1 = fit_and_test(y, X, frac)
    P2 = fit_and_test(y, X, frac, use_cvx=True)

    return P1, P2
