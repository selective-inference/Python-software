"""
This module implements the class `truncated_chi2` which 
performs (conditional) UMPU tests for Gaussians
restricted to a set of intervals.

"""
import numpy as np
from scipy.stats import chi

from truncated_bis import truncated

def pdf_chi(k, scale):
    return lambda x: chi.pdf(x/scale, k)

def cdf_chi(k, scale):
    return lambda x: chi.cdf(x/scale, k)

def quantile_chi(k, scale):
    return lambda q: scale * chi.ppf(q, k)


class truncated_chi(truncated):
    
    """
    A chi2 Gaussian distribution, truncated to
    """

    def __init__(self, intervals, k, scale):
        self._k = k
        self._scale = scale

        truncated.__init__(self,
                           intervals,
                           pdf_chi(k, scale),
                           cdf_chi(k, scale),
                           quantile_chi(k, scale))
