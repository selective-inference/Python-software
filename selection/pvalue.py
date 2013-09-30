import numpy as np
from scipy.stats import chi

def chi_pvalue(L, Mplus, Mminus, sd, k, method='MC', nsim=1000):
    if k == 1:
        H = []
    else:
        H = [0]*(k-1)
    if method == 'cdf':
        pval = (chi.cdf(Mminus / sd, k) - chi.cdf(L / sd, k)) / (chi.cdf(Mminus / sd, k) - chi.cdf(Mplus / sd, k))
    elif method == 'sf':
        pval = (chi.sf(Mminus / sd, k) - chi.sf(L / sd, k)) / (chi.sf(Mminus / sd, k) - chi.sf(Mplus / sd, k))
    elif method == 'MC':
        pval = Q_0(L / sd, Mplus / sd, Mminus / sd, H, nsim=nsim)
    elif method == 'approx':
        if Mminus < np.inf:
            num = np.log((Mminus / sd)**(k-2) * np.exp(-((Mminus/sd)**2-(L/sd)**2)/2.) - 
                         (L/sd)**(k-2))
            den = np.log((Mminus / sd)**(k-2) * np.exp(-((Mminus/sd)**2-(L/sd)**2)/2.) - 
                         (Mplus/sd)**(k-2) * np.exp(-((Mplus/sd)**2-(L/sd)**2)/2.))
            pval = np.exp(num-den)
        else:
            pval = (L/Mplus)**(k-2) * np.exp(-((L/sd)**2-(Mplus/sd)**2)/2)
    else:
        raise ValueError('method should be one of ["cdf", "sf", "MC"]')
    if pval == 1:
        pval = Q_0(L / sd, Mplus / sd, Mminus / sd, H, nsim=50000)
    if pval > 1:
        pval = 1
    return pval

def pvalue(L, Mplus, Mminus, sd, method='cdf', nsim=1000):
    return chi_pvalue(L, Mplus, Mminus, sd, 1, method=method, nsim=nsim)

def q_0(M, Mminus, H, nsim=100):
    Z = np.fabs(np.random.standard_normal(nsim))
    keep = Z < Mminus - M
    proportion = keep.sum() * 1. / nsim
    Z = Z[keep]
    if H != []:
        HM = np.clip(H + M, 0, np.inf)
        exponent = np.log(np.add.outer(Z, HM)).sum(1) - M*Z - M**2/2.
    else:
        exponent = - M*Z - M**2/2.
    C = exponent.max()

    return np.exp(exponent - C).mean() * proportion, C

def Q_0(L, Mplus, Mminus, H, nsim=100):

    exponent_1, C1 = q_0(L, Mminus, H, nsim=nsim)
    exponent_2, C2 = q_0(Mplus, Mminus, H, nsim=nsim)

    return np.exp(C1-C2) * exponent_1 / exponent_2
