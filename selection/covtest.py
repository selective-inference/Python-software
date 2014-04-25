"""
This module includes `covtest`_ that computes
either the exponential approximation from `covTest`_
the exact form of the covariance test described in 
`Spacings`_.

The covariance test itself is asymptotically exponential
(under certain conditions) and is  described in 
`covTest`_. 

Both tests mentioned above require knowledge 
(or a good estimate) of sigma, the noise variance.

This module also includes a second exact test called `reduced_covtest`_
that can use sigma but does not need it.

.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238

"""

import numpy as np
from .sample_truncnorm import sample_truncnorm_white
from .affine import constraints

def covtest(X, Y, sigma=1, exact=True):
    """
    The exact and approximate
    form of the covariance test, described
    in the `covTest`_, `Kac Rice`_ and `Spacings`_ papers.

    .. _covTest: http://arxiv.org/abs/1301.7161
    .. _Kac Rice: http://arxiv.org/abs/1308.3020
    .. _Spacings: http://arxiv.org/abs/1401.3889

    Parameters
    ----------

    X : np.float((n,p))

    Y : np.float(n)

    sigma : float (optional)
        Defaults to 1, but Type I error will be off if incorrect
        sigma is used.

    exact : bool (optional)
        If True, use the first spacings test, else use
        the exponential approximation.

    Returns
    -------

    con : `selection.constraints.constraints`_
        The constraint based on conditioning
        on the sign and location of the maximizer.

    pvalue : float
        Exact or approximate covariance test p-value.

    idx : int
        Variable achieving $\lambda_1$

    sign : int
        Sign of $X^Ty$ for variable achieving $\lambda_1$.

    """
    n, p = X.shape

    Z = np.dot(X.T, Y)
    idx = np.argsort(np.fabs(Z))[-1]
    sign = np.sign(Z[idx])

    I = np.identity(p)
    subset = np.ones(p, np.bool)
    subset[idx] = 0
    selector = np.vstack([X.T[subset],-X.T[subset]])
    selector -= (sign * X[:,idx])[None,:]

    con = constraints((selector, np.zeros(selector.shape[0])),
                      None)
    con.covariance *= sigma**2
    if exact:
        return con, con.pivot(X[:,idx] * sign, Y, 'greater'), idx, sign
    else:
        L2, L1, _, S = con.bounds(X[:,idx] * sign, Y, 'greater')
        exp_pvalue = np.exp(-L1 * (L1-L2) / S**2) # upper bound is ignored
        return con, exp_pvalue, idx, sign

def _conditional_simulation(cone_constraint, Y, 
                            ndraw=1000, burnin=1000, 
                            normalize=True, sigma=1):
    """
    Simulate from a cone instersect a sphere of radius `norm_y`.
    Assumes (implicitly) that `cone_constraint` encodes a cone constraint.

    Parameters
    ----------

    cone_constraint : `selection.affine.constraints`_

    Y : np.float
        A point in the cone. The Gibbs sampling assumes the
        point is in the cone.

    ndraw : int (optional)
        Defaults to 1000.

    burnin : int (optional)
        Defaults to 1000.

    normalize : bool (optional)
        If True, normalize sample to have Euclidean norm
        np.linalg.norm(Y)
    
    sigma : float (optional)
        Defaults to 1. Used for covariance 
        of Gaussian in the Gibbs sampler.
        
    """
    cone_matrix = cone_constraint.inequality
    
    Z = sample_truncnorm_white(cone_matrix, 
                               np.zeros(cone_matrix.shape[0]), 
                               Y, 
                               ndraw=ndraw, 
                               burnin=burnin,
                               sigma=sigma or 1.)
    if normalize:
        norm_Y = np.linalg.norm(Y)
        Z /= np.sqrt((Z**2).sum(1))[:,None]
        Z *= norm_Y
    return Z

def reduced_covtest(X, Y, ndraw=5000, burnin=1000, sigma=None):
    """
    An exact test that is more
    powerful than `covtest`_ but that requires
    sampling for the null distribution.

    This test does not require knowledge of sigma.
    
    .. _covTest: http://arxiv.org/abs/1301.7161
    .. _Kac Rice: http://arxiv.org/abs/1308.3020
    .. _Spacings: http://arxiv.org/abs/1401.3889

    Parameters
    ----------

    X : np.float((n,p))

    Y : np.float(n)

    sigma : float (optional)
        If provided, this value is used for the
        Gibbs sampler.

    exact : bool (optional)
        If True, use the first spacings test, else use
        the exponential approximation.

    Returns
    -------

    con : `selection.constraints.constraints`_
        The constraint based on conditioning
        on the sign and location of the maximizer.

    pvalue : float
        Exact p-value.

    idx : int
        Variable achieving $\lambda_1$

    sign : int
        Sign of $X^Ty$ for variable achieving $\lambda_1$.

    """

    cone, _, idx, sign = covtest(X, Y)
    A = _conditional_simulation(cone, 
                                Y,
                                ndraw=ndraw, 
                                burnin=burnin,
                                normalize=(sigma is None),
                                sigma=sigma)
    test_statistic = np.dot(A, X[:,idx]) * sign
    lam1 = np.fabs(np.dot(X.T,Y)).max()
    pvalue = (test_statistic >= lam1).sum() * 1. / ndraw
    return cone, pvalue, idx, sign

