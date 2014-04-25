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
from .affine import constraints, simulate_from_constraints
from .forward_step import forward_stepwise

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

    con : `selection.affine.constraints`_
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

    con : `selection.affine.constraints`_
        The constraint based on conditioning
        on the sign and location of the maximizer.

    pvalue : float
        Exact p-value.

    idx : int
        Variable achieving $\lambda_1$

    sign : int
        Sign of $X^Ty$ for variable achieving $\lambda_1$.

    """

    cone, _, idx, sign = covtest(X, Y, sigma=sigma or 1)
    if sigma is not None:
        cone.covariance /= sigma**2
        cone.inequality /= sigma
        cone.inequality_offset /= sigma

    Z = simulate_from_constraints(cone,
                                  Y,
                                  ndraw=ndraw,
                                  burnin=burnin,
                                  white=True)
    if sigma is None:
        norm_Y = np.linalg.norm(Y)
        Z /= np.sqrt((Z**2).sum(1))[:,None]
        Z *= norm_Y
    else:
        Z *= sigma

    test_statistic = np.dot(Z, X[:,idx]) * sign
    lam1 = np.fabs(np.dot(X.T,Y)).max()
    pvalue = (test_statistic >= lam1).sum() * 1. / ndraw
    return cone, pvalue, idx, sign

def forward_step(X, Y, sigma=1,
                 nstep=5,
                 test='reduced'):
    """
    A simple implementation of forward stepwise
    that uses the `reduced_covtest` iteratively
    after adjusting fully for the selected variable.

    This implementation is not efficient, in
    that it computes more SVDs than it really has to.

    Parameters
    ----------

    X : np.float((n,p))

    Y : np.float(n)

    nstep : int
        How many steps of forward stepwise?

    sigma : float (optional) 
        Noise level (not needed for reduced).

    test : ['reduced', 'covtest'] (optional)
        Which test to use?

    """

    FS = forward_stepwise(X, Y)
    for _ in range(nstep):
        FS.next()
    return FS
    
