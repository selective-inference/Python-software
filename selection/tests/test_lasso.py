import numpy as np
import numpy.testing.decorators as dec
from selection.lasso import lasso, data_carving, instance

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

@dec.slow
def data_carving_coverage(n=100, p=70, lam_frac=1.,
                          split_frac=0.95):
    X = np.random.standard_normal((n,p)) + 0. * np.random.standard_normal(n)[:,None]
    X -= X.mean(0)[None,:]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    sigma = 4
    beta = np.zeros(p)
    beta[:5] = 20
    mu = np.dot(X, beta) * sigma
    y = np.random.standard_normal(n) * sigma + mu
    I, L, signs = data_carving(y, X, lam_frac=lam_frac, sigma=sigma, burnin=1000, ndraw=5000,
                               center=False, scale=False)[2:]
    Xa = X[:,L.active]
    truth = np.dot(np.linalg.pinv(Xa), mu) * signs
    coverage = [(i[1] < t) * (t < i[2]) for i, t in zip(I, truth)]
    return coverage

def test_data_carving(n=100,
                      p=200,
                      s=7,
                      sigma=5,
                      rho=0.3,
                      snr=7.,
                      split_frac=0.9,
                      lam_frac=2.):

    counter = 0

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr)
        L = split_model(y, X, lam_frac=lam_frac,
                        sigma=sigma,
                        split_frac=split_frac)

        if set(range(s)).issubset(L.active):
            results, L = data_carving(y, X, lam_frac=lam_frac, 
                                      sigma=sigma,
                                      splitting=True,
                                      split_frac=split_frac)

            carve = [r[1] for r in results]
            split = [r[3] for r in results]
                return carve[s:], split[s:], carve[:s], split[:s], counter

def test_data_carving_coverage(n=200):
    C = []
    SE = np.sqrt(0.95*0.05 / n)

    while True:
        C.extend(data_carving_coverage())
        if len(C) > n:
            break

    if np.fabs(np.mean(C) - 0.95) > 2 * SE:
        raise ValueError('coverage not within 2 SE of where it should be')

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

def _instance(n=100, p=200, s=7, sigma=5, rho=0.3, snr=7):
    print 'here'
    X = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
        np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    X -= X.mean(0)[None,:]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    beta = np.zeros(p) 
    beta[:s] = snr #* (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True
    Y = (np.dot(X, beta) + np.random.standard_normal(n)) * sigma
    return X, Y, beta, np.nonzero(active)[0], sigma

