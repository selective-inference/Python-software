import numpy as np
from selection.lasso import lasso # , _howlong

def test_class(n=100, p=20):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))
    L = lasso(y,X,lam=0.5*lam_theor)
    L.fit(tol=1.e-7)
    L.form_constraints()
    C = L.constraints

    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L.y),
        L.constraints.offset)

    I = L.intervals
    P = L.active_pvalues

    return L, C, I, P


def sample_lasso(n, p, m, sigma=0.25):
    n_samples, n_features = 50, 200
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:m].fill(5)
    y = np.dot(X, beta)

    # add noise
    y += 0.01 * np.random.normal(scale = sigma, size = (n,))              

    return y, X, beta

def test_intervals(n=100, p=20, m=5, n_test = 10):
    t = []
    for i in range(n_test):
        y, X, beta = sample_lasso(n, p, m)
        las = lasso(y, X, 4., sigma = .25)
        las.fit()
        las.form_constraints()
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
    las.form_constraints()
    active = las.active_constraints
    inactive = las.inactive_constraints
    const = las.constraints



def test_pvalue():
    y, X, beta = sample_lasso(100, 50, 10)
    las = lasso(y, X, 4.)
    las.form_constraints()
    pval = las.active_pvalues


def test_nominal_intervals():
    y, X, beta = sample_lasso(100, 50, 10)
    las = lasso(y, X, 4.)
    nom_int = las.nominal_intervals




