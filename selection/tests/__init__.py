import numpy as np
from scipy.stats import t as tdist

def noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df,size=50000))
        return tdist.rvs(df, size=n) / sd_t

def equicorrelated(n=100, p=200, s=10, snr=7, sigma=5, rho=0.3,
                   df=np.inf):

    X = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
         np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    X -= X.mean(0)[None,:]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    beta = np.zeros(p) 
    beta[:s] = snr * (2 * np.random.binomial(1, 0.5, size=(s,)) - 1)
    y = (np.dot(X, beta) + noise(n, df)) * sigma

    true_active = np.zeros(p, np.bool)
    true_active[:s] = 1
    return y, X, beta, true_active
