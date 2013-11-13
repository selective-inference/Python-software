"""
Provides functions to accurately compute the pivots used in 
`selection` as well as the capability to invert these pivots.

By default, it uses `mpmath` with 100 decimal places (`mpmath.dps=100`).

"""

import numpy as np
from mpmath import mp
from scipy.optimize import bisect

DEBUG = False

def _CDF(a, b, dps=100):
    '''
    Evaluates
    $$
    \int_a^b \frac{e^{-z^2/2}}{\sqrt{2\pi}} \; dz
    $$
    using `mpmath` at `dps` decimal places.

    Parameters
    ----------

    a, b : `float`
         Limits of integration

    dps : `int`
         How many decimal places to use in `mpmath`.

    Returns
    -------

    integral : `float`
    '''
    old_dps, mp.dps = mp.dps, 100

    if a > b:
        b, a = a, b
        flip = True
    else:
        flip = False

    if a > 0:
        integral = mp.gammainc(0.5, a**2/2., b**2/2.) / (2 * mp.sqrt(mp.pi))
    elif b > 0:
        integral = (mp.gammainc(0.5, 0, b**2/2.) / (2 * mp.sqrt(mp.pi)) +
                    mp.gammainc(0.5, 0, a**2/2.) / (2 * mp.sqrt(mp.pi)))
    else:
        integral = -mp.gammainc(0.5, a**2/2., b**2/2.) / (2 * mp.sqrt(mp.pi))
    if flip:
        integral *= - 1

    mp.dps = old_dps
    return integral

def pivot(Vplus, V, Vminus, sigma, delta=0, dps=100):
    """
    Evaluates

    $$
    \frac{\Phi\left((V^--\delta)/\sigma) - \Phi\left((V-\delta)/\sigma)}
    {\Phi\left((V^--\delta)/\sigma) - \Phi\left((V^+-\delta)/\sigma)}
    $$

    Parameters
    ----------

    Vplus, V, Vminus : `float`
         Limits for pivot. Will usually satisfy $V^+ < V < V^-$.

    sigma : `float`
         Standard deviation before truncation.

    delta: `float`
         When forming intervals, this is varied to find the
         upper and lower limits.

    dps : `int`
         How many decimal places to use in `mpmath`.

    Returns
    -------

    pivot : `float`
    """
    A = _CDF((V-delta)/sigma, (Vminus-delta)/sigma, dps=dps) 
    B = _CDF((Vplus-delta)/sigma, (V-delta)/sigma, dps=dps)
    A = max(A, 0)
    B = max(B, 0)
    if (A+B)==0:
        if np.fabs(Vminus-V) > np.fabs(V-Vplus) and Vplus > 0:
            pivot = 0
        else:
            pivot = 1
    else:
        pivot = A/(A+B)
    return pivot

def solve_for_pivot(Vplus, V, Vminus, sigma, target, dps=100):
    """
    Evaluates

    $$
    \frac{\Phi\left((V^--\delta)/\sigma) - \Phi\left((V-\delta)/\sigma)}
    {\Phi\left((V^--\delta)/\sigma) - \Phi\left((V^+-\delta)/\sigma)}
    $$

    Parameters
    ----------

    Vplus, V, Vminus : `float`
         Limits for pivot. Will usually satisfy $V^+ < V < V^-$.

    sigma : `float`
         Standard deviation before truncation.

    target : `float`
         This function tries to Find $\delta$ such that 
         `pivot(Vplus, V, Vminus, sigma, delta)=target`.

    delta: `float`
         When forming intervals, this is varied to find the
         upper and lower limits.

    dps : `int`
         How many decimal places to use in `mpmath`.

    Returns
    -------

    soln : `float`

    """

    upper_guess = 1
    lower_guess = -1

    # find an upper bound
    while True:
        test = pivot(Vplus, V, Vminus, sigma, delta=upper_guess)
        if test > target:
            break
        upper_guess *= 1.5

    # find a lower bound
    while True:
        test = pivot(Vplus, V, Vminus, sigma, delta=lower_guess)
        if test < target:
            break
        lower_guess *= 1.5
    
    anon_func = lambda D: pivot(Vplus, V, Vminus, sigma, delta=D,
                                dps=dps) - target
    soln = bisect(anon_func, lower_guess, upper_guess)    
    
    if DEBUG:
        print pivot(Vplus, V, Vminus, sigma, delta=soln,
                    dps=dps), target
    return soln

def interval(Vplus, V, Vminus, sigma, upper_target=0.975, lower_target=0.025,
             dps=100):
    """
    Form an interval for the $\eta^T\mu$
    based on observing $V=\eta^TY$ and the
    constraints $V^+ \leq V \leq V^-$.

    Parameters
    ----------

    Vplus, V, Vminus : `float`
         Limits for pivot. Will usually satisfy $V^+ < V < V^-$.

    sigma : `float`
         Standard deviation before truncation.

    upper_target, lower_target : `float`
         Target for equality on each side.
    
    dps : `int`
         How many decimal places to use in `mpmath`.

    Returns
    -------

    lower, upper : `float`

    """

    lower = solve_for_pivot(Vplus, V, Vminus, sigma, lower_target, dps=dps)
    upper = solve_for_pivot(Vplus, V, Vminus, sigma, upper_target, dps=dps)
    return lower, upper

