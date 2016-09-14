import numpy as np
from nose.tools import make_decorator

def wait_for_return_value(test, max_tries=50):
    """
    Decorate a test to make it wait until the test
    returns something.
    """
    def _new_test():
        count = 0
        while True:
            v = test()
            if v is not None:
                return count, v
            count += 1
            if count >= max_tries:
                return np.inf, None
    return make_decorator(test)(_new_test)

def logistic_instance(n=100, p=200, s=7, rho=0.3, snr=14,
                      random_signs=False, df=np.inf,
                      scale=True, center=True):
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

    snr : float
        Size of each coefficient

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

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

    X = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
        np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    if center:
        X -= X.mean(0)[None,:]
    if scale:
        X /= X.std(0)[None,:]
    X /= np.sqrt(n)
    beta = np.zeros(p) 
    beta[:s] = snr 
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)

    active = np.zeros(p, np.bool)
    active[:s] = True

    eta = linpred = np.dot(X, beta) 
    pi = np.exp(eta) / (1 + np.exp(eta))

    Y = np.random.binomial(1, pi)
    return X, Y, beta, np.nonzero(active)[0]
