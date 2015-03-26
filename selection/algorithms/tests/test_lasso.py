import numpy as np
import numpy.testing.decorators as dec
from selection.algorithms.lasso import lasso, data_carving, instance, split_model

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

def test_data_carving(n=100,
                      p=200,
                      s=7,
                      sigma=5,
                      rho=0.3,
                      snr=7.,
                      split_frac=0.9,
                      lam_frac=2.,
                      ndraw=8000,
                      burnin=2000, 
                      df=np.inf,
                      coverage=0.90):

    counter = 0

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr, 
                                             df=df)
        mu = np.dot(X, beta)
        L, stage_one = split_model(y, X, 
                        sigma=sigma,
                        lam_frac=lam_frac,
                        split_frac=split_frac)[:2]

        if set(range(s)).issubset(L.active):
            results, L = data_carving(y, X, lam_frac=lam_frac, 
                                      sigma=sigma,
                                      stage_one=stage_one,
                                      splitting=True, 
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      coverage=coverage)

            carve = [r[1] for r in results]
            split = [r[3] for r in results]

            Xa = X[:,L.active]
            truth = np.dot(np.linalg.pinv(Xa), mu) 
            print np.dot(np.linalg.pinv(Xa), y)[:s]

            split_coverage = []
            carve_coverage = []
            for result, t in zip(results, truth):
                _, _, ci, _, si = result
                print si, ci, t
                carve_coverage.append((ci[0] < t) * (t < ci[1]))
                split_coverage.append((si[0] < t) * (t < si[1]))

            return carve[s:], split[s:], carve[:s], split[:s], carve_coverage, split_coverage

@dec.slow
def test_data_carving_coverage(n=200, coverage=0.8):
    C = []
    SE = np.sqrt(coverage * (1 - coverage) / n)

    while True:
        C.extend(data_carving_coverage()[-1])
        if len(C) > n:
            break

    if np.fabs(np.mean(C) - coverage) > 3 * SE:
        raise ValueError('coverage not within 3 SE of where it should be')

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

