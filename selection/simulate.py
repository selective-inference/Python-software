import numpy as np
from lasso import interval_constraint_linf, interval_constraint_max
from scipy.stats import norm as ndist
from pvalue import pvalue as lasso_pvalue

import statsmodels.api as sm # recommended import according to the docs

import matplotlib.pyplot as plt

def simulate_from_bounds(lower_bound, upper_bound, XtX, S=1):
    upper_bound = np.asarray(upper_bound)
    lower_bound = np.asarray(lower_bound)

    U, D, V = np.linalg.svd(XtX)
    sqroot = U * np.sqrt(D)

    p = lower_bound.shape[0]
    if upper_bound.shape[0] != p:
        raise ValueError('bounds should be 1d arrays of same shape')

    while True:
        Z = np.dot(sqroot, np.random.standard_normal(p) * S)
        if np.all(Z >= lower_bound) and np.all(Z <= upper_bound):
            return Z

def simulate_linf(lower_bound, upper_bound, sigma_epsilon=1, n=20):

    p = np.asarray(lower_bound).shape[0]
    X = np.random.standard_normal((n,p)) + 0.5 * np.random.standard_normal(n)[:,None]
    X -= X.mean(0)
    X /= np.sqrt(n)
    XtX = np.dot(X.T, X)
    B = simulate_from_bounds(lower_bound, upper_bound, XtX, S=sigma_epsilon)

    S2 = np.zeros((2*p,2*p))
    S2[:p,:p] = XtX
    S2[p:,p:] = XtX
    S2[:p,p:] = -XtX
    S2[p:,:p] = -XtX
    upper_bound2 = np.hstack([upper_bound,-lower_bound])
    lower_bound2 = np.hstack([lower_bound,-upper_bound])
    B2 = np.hstack([B,-B])

    L, Vplus, Vminus, var_star, mean_star = \
        interval_constraint_linf(B, XtX, np.zeros(p), 
                                 lower_bound=lower_bound,
                                 upper_bound=upper_bound)


    L2, Vplus2, Vminus2, var_star2, mean_star2 = \
        interval_constraint_max(B2, S2, np.zeros(2*p), 
                                lower_bound=lower_bound2,
                                upper_bound=upper_bound2)
    
    if ((np.fabs(Vminus - Vminus2) > 0.01) or (np.fabs(Vplus - Vplus2) > 0.01)
        or (np.fabs(L - L2) > 0.01)):
        stop

    sigma = np.sqrt(var_star) * sigma_epsilon
    pval = (ndist.sf((Vminus-mean_star) / sigma) - ndist.sf((L-mean_star)/sigma)) / (ndist.sf((Vminus-mean_star) / sigma) - ndist.sf((Vplus-mean_star)/sigma))
    if np.isnan(pval):
        pval = lasso_pvalue(L, Vplus, Vminus, sigma, method='MC', nsim=100000)
    return pval, (Vplus-mean_star) / sigma, (L-mean_star) / sigma, (Vminus-mean_star) / sigma

def simulate_max(lower_bound, upper_bound, sigma_epsilon=1, n=20):

    p = np.asarray(lower_bound).shape[0]
    X = np.random.standard_normal((n,p)) + 0.5 * np.random.standard_normal(n)[:,None]
    X -= X.mean(0)
    X /= np.sqrt(n)
    XtX = np.dot(X.T, X)
    B = simulate_from_bounds(lower_bound, upper_bound, XtX, S=sigma_epsilon)

    L, Vplus, Vminus, var_star, mean_star = \
        interval_constraint_max(B, XtX, np.zeros(p), 
                                lower_bound=lower_bound,
                                upper_bound=upper_bound)

    sigma = np.sqrt(var_star) * sigma_epsilon
    pval = (ndist.sf((Vminus-mean_star) / sigma) - ndist.sf((L-mean_star)/sigma)) / (ndist.sf((Vminus-mean_star) / sigma) - ndist.sf((Vplus-mean_star)/sigma))
    if np.isnan(pval):
        pval = lasso_pvalue(L, Vplus, Vminus, sigma, method='MC', nsim=100000)
    return pval, (Vplus-mean_star) / sigma, (L-mean_star) / sigma, (Vminus-mean_star) / sigma

def test_linf():

    p = 15
    L = np.array([-1]*p) + 0.1 * np.random.sample(p)
    U = np.array([1]*p) + 0.1 * np.random.sample(p)
    sigma = 0.9

    P = []
    for i in range(20000):
        V = simulate_linf(L, U, sigma_epsilon=sigma)
        P.append(V[0])
        if i % 50 == 0:
            Q = np.array(P)
            Q = Q[(Q > 0)*(Q<1)]
            print np.mean(Q), np.std(Q)

    P = np.array(P)
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    plt.clf()
    y = ecdf(x)
    plt.plot([0,1],[0,1], '--', linewidth=1, color='black')
    plt.step(x, y, linewidth=2)

def test_max():

    p = 15
    L = np.array([-1]*p) + 0.1 * np.random.sample(p)
    U = np.array([1]*p) + 0.1 * np.random.sample(p)
    sigma = 0.9

    P = []
    for i in range(10000):
        P.append(simulate_max(L, U, sigma_epsilon=sigma)[0])
        if i % 50 == 0:
            Q = np.array(P)
            Q = Q[(Q > 0)*(Q<1)]
            print np.mean(Q), np.std(Q)

    P = np.array(P)
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    plt.clf()
    y = ecdf(x)
    plt.plot([0,1],[0,1], '--', linewidth=1, color='black')
    plt.step(x, y, linewidth=2)
