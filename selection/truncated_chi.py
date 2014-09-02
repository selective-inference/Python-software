"""
This module implements the class `truncated_chi2` which 
performs (conditional) UMPU tests for Gaussians
restricted to a set of intervals.

"""
import numpy as np
import mpmath as mp
from scipy.stats import chi

import rpy2.robjects as robjects

# import truncated_bis
# truncated_bis = reload(truncated_bis)
from truncated_bis import truncated

def pdf_chi(k, scale):
    return lambda x: chi.pdf(x/scale, k)

def cdf_chi(k, scale):
    return lambda x: chi.cdf(x/scale, k)

def quantile_chi(k, scale):
    return lambda q: scale * chi.ppf(q, k)

def sf_chi2(k, scale):
    def sf(a, b=np.inf, dps=15):
        dps_temp = mp.mp.dps
        mp.mp.dps = dps
        sf = mp.gammainc(1./2 * k, 
                         1./2*(a/scale), 
                         1./2*(b/scale), 
                         regularized=True)
        mp.mp.dps = dps_temp
        return sf

    return sf
    

def sf_chi(k, scale):

    def sf(a, b=np.inf, dps=15):
        dps_temp = mp.mp.dps
        mp.mp.dps = dps
        sf = mp.gammainc(1./2 * k, 
                         1./2*((a/scale)**2), 
                         1./2*((b/scale)**2), 
                         regularized=True)
        mp.mp.dps = dps_temp
        return sf

    return sf



class truncated_chi(truncated):
    
    """
    A chi Gaussian distribution, truncated to
    """

    def __init__(self, intervals, k, scale=1):
        """
        >>> distr = truncated_chi([(1., 3.)], 3, 1.)
        """
        self._k = k
        self._scale = scale

        truncated.__init__(self,
                           intervals,
                           pdf_chi(k, scale),
                           cdf_chi(k, scale),
                           sf_chi(k, scale),
                           quantile_chi(k, scale))



class truncated_chi2(truncated):
    
    """
    A chi Gaussian distribution, truncated to
    """

    def __init__(self, intervals, k, scale=1):
        """
        >>> distr = truncated_chi([(1., 3.)], 3, 1.)
        """
        self._k = k
        self._scale = scale

        truncated.__init__(self,
                           intervals,
                           pdf_chi(k, scale),
                           cdf_chi(k, scale),
                           sf_chi2(k, scale),
                           quantile_chi(k, scale))



import doctest
doctest.testmod()

