import numpy as np
from selection.algorithms.lasso import instance as lasso_instance
from selection.algorithms.randomized import logistic_instance, randomized_lasso, randomized_logistic

def test_logistic(n=200, p=30, burnin=2000, ndraw=8000,
                  compute_interval=False,
                  sandwich=True,
                  s=6):

    X, y, beta, lasso_active = logistic_instance(n=n, p=p, snr=10, s=s, scale=False, center=False,
                                                 rho=0.1)
    n, p = X.shape

    lam = 0.6 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0)) / 2

    L = randomized_logistic(y, X, lam, (True, 0.4 * np.diag(np.sqrt(np.diag(np.dot(X.T, X))))),
                            sandwich=sandwich)
    L.fit()

    if (set(range(s)).issubset(L.active) and 
        L.active.shape[0] > s):
        L.constraints.mean[:p] = 0 * L.unbiased_estimate

        v = np.zeros_like(L.active)
        v[s] = 1.
        P0, interval = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                         compute_interval=compute_interval)
        target = (beta[L.active]*v).sum()
        estimate = (L.unbiased_estimate[:L.active.shape[0]]*v).sum()
        low, hi = interval

        v = np.zeros_like(L.active)
        v[0] = 1.
        PA, _ = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                  compute_interval=compute_interval)

        return P0, PA, L

def test_gaussian(n=200, p=30, burnin=2000, ndraw=8000,
                  compute_interval=False,
                  s=6):

    X, y, beta, lasso_active, sigma = lasso_instance(n=n, 
                                                     p=p)
    n, p = X.shape

    lam = sigma * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    L = randomized_lasso(y, X, lam, (True, 0.4 * np.diag(np.sqrt(np.diag(np.dot(X.T, X))))),
                         sandwich=False)
    L.fit()

    L = randomized_lasso(y, X, lam, (True, 0.4 * np.diag(np.sqrt(np.diag(np.dot(X.T, X))))),
                         sandwich=True)
    L.fit()
    if (set(range(s)).issubset(L.active) and 
        L.active.shape[0] > s):
        L.constraints.mean[:p] = 0 * L.unbiased_estimate

        v = np.zeros_like(L.active)
        v[s] = 1.
        P0, interval = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                         compute_interval=compute_interval)
        target = (beta[L.active]*v).sum()
        estimate = (L.unbiased_estimate[:L.active.shape[0]]*v).sum()
        low, hi = interval

        v = np.zeros_like(L.active)
        v[0] = 1.
        PA, _ = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                  compute_interval=compute_interval)

        return P0, PA, L
