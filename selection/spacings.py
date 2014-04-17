import numpy as np
from .constraints import constraints

def covtest(X, Y, sigma=1):
    """
    The exact form of the covariance test, described
    in the `Kac Rice`_ and `Spacings`_ papers.

    .. _Kac Rice: http://arxiv.org/abs/1308.3020
    .. _Spacings: http://arxiv.org/abs/1401.3889

    Parameters
    ----------

    X : np.float((n,p))

    Y : np.float(n)

    sigma : float

    Returns
    -------

    con : `selection.constraints.constraints`_
        The constraint based on conditioning
        on the sign and location of the maximizer.

    pvalue : float
        Exact covariance test p-value.

    """
    n, p = X.shape

    Z = np.dot(X.T, Y)
    idx = np.argsort(np.fabs(Z))[-1]
    sign = np.sign(Z[idx])

    I = np.identity(p)
    subset = np.ones(p, np.bool)
    subset[idx] = 0
    selector = np.dot(np.vstack([I[subset],-I[subset]]), X.T)
    selector -= (sign * X[:,idx])[None,:]

    con = constraints((selector, np.zeros(selector.shape[0])),
                      None)

    return con, con.pivot(X[:,idx] * sign, Y, 'greater')



