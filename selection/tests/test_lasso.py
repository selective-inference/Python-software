import numpy as np
from selection.lasso import lasso # , _howlong

def test_class(n=100, p=20, frac=0.9):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    L = lasso(y,X,frac=frac)
    L.fit(tol=1.e-7)
    L.form_constraints()
    C = L.constraints

    np.testing.assert_array_less( \
        np.dot(L.constraints.inequality, L.y),
        L.constraints.inequality_offset)

    I = L.intervals
    P = L.active_pvalues

    return L.centered_test, L.basic_test, L, C, I, P


def sample_lasso(n, p, m, sigma=0.25):
    n_samples, n_features = 50, 200
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:m].fill(5)
    y = np.dot(X, beta)

    # add noise
    y += 0.01 * np.random.normal(scale = sigma, size = (n,))              

    return y, X, beta

def test_intervals(n, p, m, n_test = 10):
    t = []
    for i in range(n_test):
        y, X, beta = sample_lasso(n, p, m)
        las = lasso(y, X, 4., sigma = .25)
        las.fit()
        intervals = las.intervals
        t.append([(beta[I[0]], I[3]) for I in intervals])
    return t
        
    
def test_soln():
    y, X, bet = sample_lasso(100, 50, 10)
    las = lasso(y, X, 4.)
    beta2 = las.soln



def test_constraints():
    y, X, beta = sample_lasso(100, 50, 10)
    las = lasso(y, X, 4.)
    las.fit()
    active = las.active_constraints
    inactive = las.inactive_constraints
    const = las.constraints



def test_pvalue():
    y, X, beta = sample_lasso(100, 50, 10)
    las = lasso(y, X, 4.)
    pval = las.active_pvalues()



def test_nominal_intervals():
    y, X, beta = sample_lasso(100, 50, 10)
    las = lasso(y, X, 4.)
    nom_int = las.nominal_intervals





def sample_test(y, X, lam):
    las = lasso(y, X, lam)
    beta2 = las.soln
    return beta2



    
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
