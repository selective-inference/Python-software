import numpy as np
import pandas as pd

from scipy.stats import t as tdist

# def design(n, p, rho, equi_correlated):
#     if equi_correlated:
#         X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
#              np.sqrt(rho) * np.random.standard_normal(n)[:, None])
#     else:
#         def AR1(rho, p):
#             idx = np.arange(p)
#             cov = rho ** np.abs(np.subtract.outer(idx, idx))
#             return cov, np.linalg.cholesky(cov)

#         sigmaX, cholX = AR1(rho=rho, p=p)
#         X = np.random.standard_normal((n, p)).dot(cholX.T)
#         # X = np.random.multivariate_normal(mean=np.zeros(p), cov = sigmaX, size = (n,))
#         # print(X.shape)
#     return X

def gaussian_instance(n=100, p=200, s=7, sigma=5, rho=0.3, signal=7,
                      random_signs=False, df=np.inf,
                      scale=True, center=True,
                      equi_correlated=True):


    """
    A testing instance for the LASSO.
    If equi_correlated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equi_correlated is False design is auto-regressive.
    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.
    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of features
    s : int
        True sparsity
    sigma : float
        Noise level
    rho : float
        Equicorrelation value (must be in interval [0,1])

    signal : float
        Size of each coefficient

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    equi_correlated: bool
        If true, design in equi-correlated,
        Else design is AR.

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.
    """
    X=design(n,p, rho, equi_correlated)


    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None,:] * np.sqrt(n))
    beta = np.zeros(p) 
    beta[:s] = signal 

    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise(n, df)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma

_cholesky_factors = {} # should we store them?

def _AR_cov(p, rho=0.25):
    idx = np.arange(p)
    return rho**np.fabs(np.subtract.outer(idx, idx))

def _AR_sqrt_cov(p, rho=0.25):
    idx = np.arange(p)
    C = rho**np.fabs(np.subtract.outer(idx, idx))
    return np.linalg.cholesky(C)


def AR_instance(n=2000, p=2500, s=30, sigma=2, rho=0.25, signal=4.5):
    """
    Used to compare to Barber and Candes high-dim knockoff.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    sigma : float
        Noise level

    rho : float
        AR(1) parameter.

    signal : float
        Size of each coefficient

    """

    if (rho, p) not in _cholesky_factors.keys():
        _cholesky_factors[(rho, p)] = _AR_sqrt_cov(p, rho)
    _sqrt_cov = _cholesky_factors[(rho, p)]

    X = np.random.standard_normal((n, p)).dot(_sqrt_cov.T)

    X /= (np.sqrt((X**2).sum(0))) # like normc
    beta = np.zeros(p)
    beta[:s] = signal * (2 * np.random.binomial(1, 0.5, size=(s,)) - 1) 
    np.random.shuffle(beta)

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma
    true_active = np.nonzero(beta != 0)[0]
    return X, Y, beta * sigma, true_active, sigma

def logistic_instance(n=100, p=200, s=7, rho=0.3, signal=14,
                      random_signs=False, 
                      scale=True, center=True, equi_correlated=True):
    """
    A testing instance for the LASSO.
    Design is equi-correlated in the population,
    normalized to have columns of norm 1.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    rho : float
        Equicorrelation value (must be in interval [0,1])

    signal : float
        Size of each coefficient

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    """

    X= design(n,p, rho, equi_correlated)

    if center:
        X -= X.mean(0)[None,:]
    if scale:
        X /= X.std(0)[None,:]
    X /= np.sqrt(n)
    beta = np.zeros(p) 
    beta[:s] = signal 
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)

    active = np.zeros(p, np.bool)
    active[:s] = True

    eta = linpred = np.dot(X, beta) 
    pi = np.exp(eta) / (1 + np.exp(eta))

    Y = np.random.binomial(1, pi)
    return X, Y, beta, np.nonzero(active)[0]

def HIV_NRTI(drug='3TC', 
             standardize=True, 
             datafile=None,
             min_occurrences=11):
    """
    Download 

        http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt

    and return the data set for a given NRTI drug.

    The response is an in vitro measurement of log-fold change 
    for a given virus to that specific drug.

    Parameters
    ----------

    drug : str (optional)
        One of ['3TC', 'ABC', 'AZT', 'D4T', 'DDI', 'TDF']

    standardize : bool (optional)
        If True, center and scale design X and center response Y.

    datafile : str (optional)
        A copy of NRTI_DATA above.

    min_occurrences : int (optional)
        Only keep positions that appear
        at least a minimum number of times.
        

    """

    if datafile is None:
        datafile = "http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt"
    NRTI = pd.read_table(datafile, na_values="NA")

    NRTI_specific = []
    NRTI_muts = []
    mixtures = np.zeros(NRTI.shape[0])
    for i in range(1,241):
        d = NRTI['P%d' % i]
        for mut in np.unique(d):
            if mut not in ['-','.'] and len(mut) == 1:
                test = np.equal(d, mut)
                if test.sum() >= min_occurrences:
                    NRTI_specific.append(np.array(np.equal(d, mut))) 
                    NRTI_muts.append("P%d%s" % (i,mut))

    NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)

    X_NRTI = np.array(NRTI_specific, np.float)
    Y = NRTI[drug] # shorthand
    keep = ~np.isnan(Y).astype(np.bool)
    X_NRTI = X_NRTI[np.nonzero(keep)]; Y=Y[keep]
    Y = np.array(np.log(Y), np.float); 

    if standardize:
        Y -= Y.mean()
        X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
    return X_NRTI, Y, np.array(NRTI_muts)
