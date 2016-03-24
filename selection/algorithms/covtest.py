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

This module also includes a second exact test called `selected_covtest`_
that can use sigma but does not need it.

.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238

"""

import numpy as np
from scipy.special import ndtr, ndtri

from ..constraints.affine import constraints, sample_from_constraints, gibbs_test
from ..distributions.discrete_family import discrete_family

def covtest(X, Y, sigma=1, exact=True,
            covariance=None):
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

    covariance : np.array (optional)
        If None, defaults to identity.

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

    if covariance is None:
        covariance = np.identity(n)

    Z = np.dot(X.T, Y)
    idx = np.argsort(np.fabs(Z))[-1]
    sign = np.sign(Z[idx])

    I = np.identity(p)
    subset = np.ones(p, np.bool)
    subset[idx] = 0
    selector = np.vstack([X.T[subset],-X.T[subset],-sign*X[:,idx]])
    selector -= (sign * X[:,idx])[None,:]

    con = constraints(selector, np.zeros(selector.shape[0]),
                      covariance=covariance)
    con.covariance *= sigma**2
    if exact:
        return con, con.pivot(X[:,idx] * sign, Y, alternative='greater'), idx, sign
    else:
        L2, L1, _, S = con.bounds(X[:,idx] * sign, Y)
        exp_pvalue = np.exp(-L1 * (L1-L2) / S**2) # upper bound is ignored
        return con, exp_pvalue, idx, sign

def selected_covtest(X, Y, ndraw=5000, burnin=2000, sigma=None,
                    covariance=None):
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

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    sigma : float (optional)
        If provided, this value is used for the
        Gibbs sampler.

    covariance : np.float (optional)
        Optional covariance for cone constraint.
        Will be scaled by sigma if it is not None.

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

    cone, _, idx, sign = covtest(X, Y, sigma=sigma or 1,
                                 covariance=covariance)

    if sigma is None:
        pvalue, Z, W = gibbs_test(cone, Y, X[:,idx] * sign,
                                  ndraw=ndraw,
                                  burnin=burnin,
                                  sigma_known=False, 
                                  alternative='greater')[:3]
    else:
        val = np.sum((X[:,idx] * Y) * sign)
        family = _covtest_sampler(cone, X[:,idx] * sign,
                                  sigma, ndraw=ndraw, 
                                  mu = val * X[:,idx] * sign 
                                  / np.linalg.norm(X[:,idx])**2)
        pvalue = family.ccdf(-val / sigma**2, val)
    return cone, pvalue, idx, sign

def _covtest_sampler(cone, eta, sigma, ndraw=1000, mu=None):
    """
    Due to special strucutre of covtest cone constraint, sampling
    is easy with importance weights.
    """
    n = eta.shape[0]
    eta_n = eta / np.linalg.norm(eta)

    results = []
    weights = []

    if mu is None:
        mu = np.zeros(n)

    for _ in range(ndraw):
        Y0 = np.random.standard_normal(n) * sigma + mu
        mu_eta = (mu * eta_n).sum()
        Y0 -= (Y0 * eta_n).sum() * eta_n
        L, _, U = cone.bounds(eta_n, Y0)[:3]
        cdfL = ndtr(-(L - mu_eta) / sigma)
        cdfU = ndtr(-(U - mu_eta) / sigma)
        unif = np.random.sample() * (cdfU - cdfL) + cdfL
        if unif < 0.5:
            tnorm = ndtri(unif) * sigma
        else:
            tnorm = -ndtri(1-unif) * sigma
        tnorm = -tnorm
        results.append(np.sum(eta * (Y0 + (tnorm + mu_eta) * eta_n)))
        weights.append(np.fabs(cdfL - cdfU))
                           
    family = discrete_family(results, weights)
    return family
