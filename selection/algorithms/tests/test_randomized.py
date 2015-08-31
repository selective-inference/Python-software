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

def compare_sandwich():
    import matplotlib.pyplot as plt

    P0 = {}
    PA = {}

    def nonnan(v):
        v = np.asarray(v)
        return (~np.isnan(v)).sum()

    for i in range(2000):
        for sandwich in [True,False]:
            P0.setdefault(sandwich, [])
            PA.setdefault(sandwich, [])
            R = test_logistic(burnin=5000, ndraw=10000, sandwich=sandwich)
            if R is not None:
                P0[sandwich].append(R[0])
                PA[sandwich].append(R[1])
        if ((nonnan(P0[True]) > 200)
            and (nonnan(P0[False]) > 200)
            and (nonnan(PA[False]) > 200)
            and (nonnan(PA[True]) > 200)):
            break
        print nonnan(P0[True]), nonnan(P0[False]), nonnan(PA[True]), nonnan(PA[False])

    def nanclean(v):
        v = np.array(v)
        return v[~np.isnan(v)]

    plt.clf()
    U = np.linspace(0,1, 101)

    plt.plot(U, sm.distributions.ECDF(nanclean(PA[False]))(U), 'r--', label='Parametric', linewidth=3)
    plt.plot(U, sm.distributions.ECDF(nanclean(PA[True]))(U), 'r:', label='Sandwich', linewidth=3)
    plt.plot(U, sm.distributions.ECDF(nanclean(P0[False]))(U), 'k--', linewidth=3)
    plt.plot(U, sm.distributions.ECDF(nanclean(P0[True]))(U), 'k:', linewidth=3)
    plt.legend(loc='lower right')
    plt.savefig('compare.pdf')
