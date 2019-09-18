"""

This module contains functions needed to evaluate post selection
p-values for non polyhedral selection procedures through a variety of means.

These p-values appear for the group LASSO global null test as well as the nuclear norm
p-value test.

They are described in the `Kac Rice`_ paper.

.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238


"""

import numpy as np
from scipy.stats import chi

from scipy.stats import norm as ndist, truncnorm 
from scipy.integrate import quad

from mpmath import mp
mp.dps = 80

def norm_q(prob):
    r"""
    A multi-precision calculation of the
    standard normal quantile function:

    .. math::

       \int_{-\infty}^{q(p)} \frac{e^{-z^2/2}}{\sqrt{2\pi}} \; dz = p

    where $p$ is `prob`.

    Parameters
    ----------

    prob : float

    Returns
    -------

    quantile : float

    """
    return np.array(mp.erfinv(2*prob-1)*mp.sqrt(2))


def norm_pdf(observed):
    r"""
    A multi-precision calculation of the
    standard normal density function:

    .. math::

       \frac{e^{-T^2/2}}{\sqrt{2\pi}}

    where `T` is observed.

    Parameters
    ----------

    observed : float

    Returns
    -------

    density : float

    """
    return np.array(mp.npdf(observed))

def norm_interval(lower, upper):
    r"""
    A multiprecision evaluation of

    .. math::

        \Phi(U) - \Phi(L)

    Parameters
    ----------

    lower : float
        The lower limit $L$

    upper : float
        The upper limit $U$

    """
    if lower > 0 and upper > 0:
        return mp.ncdf(-lower) - mp.ncdf(-upper)
    else:
        return mp.ncdf(upper) - mp.ncdf(lower)

def truncnorm_cdf(observed, lower, upper):
    r"""
    Compute the truncated normal 
    distribution function.

    .. math::

        \frac{\Phi(U) - \Phi(T)}{\Phi(U) - \Phi(L)}

    where $T$ is `observed`, $L$ is `lower_bound` and $U$ is `upper_bound`.

    Parameters
    ----------

    observed : float

    lower : float

    upper : float

    Returns
    -------

    P : float

    """
    x, a, b = observed, lower, upper

    x = max(x, a)
    x = min(x, b)

    #a = max(min(a, 5), -5)
    #b = max(min(b, 5), -5)
    #x = max(min(x, 5), -5)
    #if a==b:
    #    return 1.

    if a > 0 and b > 0:
        Fx, Fa, Fb = mp.ncdf(-x), mp.ncdf(-a), mp.ncdf(-b)
        return float(( Fa - Fx ) / (Fa - Fb))
    else:
        Fx, Fa, Fb = mp.ncdf(x), mp.ncdf(a), mp.ncdf(b)
        return float( ( Fx - Fa ) / ( Fb - Fa ) )


def chi_pvalue(observed, lower_bound, upper_bound, sd, df, method='MC', nsim=1000):
    r"""

    Compute a truncated $\chi$ p-value based on the 
    conditional survival function. 

    Parameters
    ----------

    observed : float

    lower_bound : float

    upper_bound : float

    sd : float
        Standard deviation.

    df : float
        Degrees of freedom.

    method: string
        One of ['MC', 'cdf', 'sf']

    Returns
    -------

    pvalue : float

    Notes
    -----

    Let $T$ be `observed`, $L$ be `lower_bound` and $U$ be `upper_bound`,
    and $\sigma$ be `sd`.
    The p-value, for $L \leq T \leq U$ is

    .. math::

         \frac{P(\chi^2_k / \sigma^2 \geq T^2) - P(\chi^2_k / \sigma^2 \geq U^2)}
         {P(\chi^2_k / \sigma^2 \geq L^2) - P(\chi^2_k / \sigma^2 \geq U^2)} 

    It can be computed using `scipy.stats.chi` either its `cdf` (distribution 
    function) or `sf` (survival function) or evaluated
    by Monte Carlo if method is `MC`.

    """

    L, T, U = lower_bound, observed, upper_bound # shorthand

    if method == 'cdf':
        pval = ((chi.cdf(U / sd, df) - chi.cdf(T / sd, df)) / 
                (chi.cdf(U / sd, df) - chi.cdf(L / sd, df)))
    elif method == 'sf':
        pval = ((chi.sf(U / sd, df) - chi.sf(T / sd, df)) / 
                (chi.sf(U / sd, df) - chi.sf(L / sd, df)))
    elif method == 'MC':
        if df == 1:
            H = []
        else:
            H = [0]*(df-1)
        pval = general_pvalue(T / sd, L / sd, U / sd, H, nsim=nsim)
    else:
        raise ValueError('method should be one of ["cdf", "sf", "MC"]')
    if pval == 1: # the distribution functions may have failed -- use MC
        pval = general_pvalue(T / sd, L / sd, U / sd, H, nsim=50000)
    if pval > 1:
        pval = 1
    return pval

def gauss_poly(lower_bound, upper_bound, curvature, nsim=100):
    r"""
    Computes the integral of a polynomial times the 
    standard Gaussian density over an interval.

    Introduced in `Kac Rice`_, display (33) of v2.

    Parameters
    ----------

    lower_bound : float

    upper_bound : float

    curvature : np.array
        A diagonal matrix related to curvature.
        It is assumed that `curvature + lower_bound I` is non-negative definite.

    nsim : int
        How many draws from $N(0,1)$ should we use?

    Returns
    -------

    integral : float

    Notes
    -----

    The return value is a Monte Carlo estimate of

    .. math::

        \int_{L}^{U} \det(\Lambda + z I)
        \frac{e^{-z^2/2\sigma^2}}{\sqrt{2\pi\sigma^2}} \, dz

    where $L$ is `lower_bound`, $U$ is `upper_bound` and $\Lambda$ is the
    diagonal matrix `curvature`.

    """

    T, H = lower_bound, curvature
    proportion = ndist.cdf(upper_bound - T) - ndist.cdf(0)
    U = np.random.sample(nsim)  
    Z = ndist.ppf(U * proportion + ndist.cdf(0))
    if H != []:
        HT = H + T 
        exponent = np.log(np.fabs(np.add.outer(Z, HT))).sum(1) - T*Z - T**2/2.
    else:
        exponent = - T*Z - T**2/2.
    C = exponent.max()

    return np.exp(exponent - C).mean() * proportion, C

def general_pvalue(observed, lower_bound, upper_bound, curvature, nsim=100):

    r"""
    Computes the integral of a polynomial times the 
    standard Gaussian density over an interval.

    Introduced in `Kac Rice`_, display (35) of v2.

    Parameters
    ----------

    observed : float

    lower_bound : float

    upper_bound : float

    curvature : np.array
        A diagonal matrix related to curvature.
        It is assumed that `curvature + lower_bound I` is non-negative definite.

    nsim : int
        How many draws from $N(0,1)$ should we use?

    Returns
    -------

    integral : float

    Notes
    -----

    The return value is a Monte Carlo estimate of

    .. math::

        \frac{\int_{T}^{U} \det(\Lambda + z I)
        \frac{e^{-z^2/2\sigma^2}}{\sqrt{2\pi\sigma^2}} \, dz}
        {\int_{L}^{U} \det(\Lambda + z I)
        \frac{e^{-z^2/2\sigma^2}}{\sqrt{2\pi\sigma^2}} \, dz}

    where $T$ is `observed`, $L$ is `lower_bound`, 
    $U$ is `upper_bound` and $\Lambda$ is the
    diagonal matrix `curvature`.

    """

    exponent_1, C1 = gauss_poly(observed, upper_bound, curvature, nsim=nsim)
    exponent_2, C2 = gauss_poly(lower_bound, upper_bound, curvature, nsim=nsim)

    return np.exp(C1-C2) * exponent_1 / exponent_2


class SelectionInterval(object):
    """
    Compute a selection interval for
    a Gaussian truncated to an interval.
    """

    def __init__(self, lower_bound, observed, upper_bound, sigma):
        self.lower_bound, self.observed, self.upper_bound, self.sigma = lower_bound, observed, upper_bound, sigma


    def pivot(self, exp):
        L, obs, U, sd = self.lower_bound, self.observed, self.upper_bound, self.sigma 
        return truncnorm_cdf((obs-exp)/sd, (L-exp)/sd, (U-exp)/sd)

    def conf_int(self, lb, ub, alpha=0.05):
        F = lambda exp: self.pivot(exp)         
        L = _find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        U = _find_root(F, 0.5 * alpha, L, ub)
        return L, U

def _find_root(f, y, lb, ub, tol=1e-6):
    """
    searches for solution to f(x) = y in (lb, ub), where 
    f is a monotone decreasing function
    """       
    
    # make sure solution is in range
    a, b   = lb, ub
    fa, fb = f(a), f(b)
    
    # assume a < b
    if fa > y and fb > y:
        while fb > y : 
            b, fb = b + (b-a), f(b + (b-a))
    elif fa < y and fb < y:
        while fa < y : 
            a, fa = a - (b-a), f(a - (b-a))
    
    # determine the necessary number of iterations
    max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )
    
    # bisect (slow but sure) until solution is obtained
    for _ in xrange(max_iter):
        c, fc  = (a+b)/2, f((a+b)/2)
        if fc > y: a = c
        elif fc < y: b = c

    return c
