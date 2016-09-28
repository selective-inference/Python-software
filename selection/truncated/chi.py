"""
This module implements the class `truncated_chi2` which 
performs (conditional) UMPU tests for Gaussians
restricted to a set of intervals.

"""
import numpy as np
import mpmath as mp
from scipy.stats import chi, chi2

from .base import truncated, find_root

class truncated_chi(truncated):

    """
    >>> from selection.constraints.intervals import intervals
    >>> I = intervals.intersection(intervals((-1, 6)),
    ...                            intervals(( 0, 7)),
    ...                           ~intervals((1, 4)))
    >>> distr = truncated_chi(I, 3, 2.)
    >>> print(abs(distr.cdf(distr.quantile(0.9)) - 0.9) < 0.01)
    True
    """
    def __init__(self, I, k, scale = 1.):
        """
        Create a new object for a truncated_chi distribution

        Parameters
        ----------
        I : intervals
            The intervals the distribution is truncated to

        k : int
            Number of degree of freedom of the distribution

        scale : float
            The distribution is \sim scale * \chi_k

        
        """

        self._k = k
        self._scale = scale
        truncated.__init__(self, I)

    def _cdf_notTruncated(self, a, b, dps):
        """
        Compute the probability of being in the interval (a, b)
        for a variable with a chi distribution (not truncated)
        
        Parameters
        ----------
        a, b : float
            Bounds of the interval. Can be infinite.

        dps : int
            Decimal precision (decimal places). Used in mpmath

        Returns
        -------
        p : float
            The probability of being in the intervals (a, b)
            P( a < X < b)
            for a non truncated variable

        """
        scale = self._scale
        k = self._k

        dps_temp = mp.mp.dps
        mp.mp.dps = dps

        a = max(0, a)
        b = max(0, b)

        sf = mp.gammainc(1./2 * k, 
                         1./2*((a/scale)**2), 
                         1./2*((b/scale)**2), 
                         regularized=True)
        mp.mp.dps = dps_temp
        return sf

    def _pdf_notTruncated(self, z, dps):
        scale = self._scale
        k = self._k
        dps = self._dps

        return chi.pdf(z/scale, k)

    def _quantile_notTruncated(self, q, tol=1.e-6):
        """
        Compute the quantile for the non truncated distribution

        Parameters
        ----------
        q : float
            quantile you want to compute. Between 0 and 1

        tol : float
            precision for the output

        Returns
        -------
        x : float
            x such that P(X < x) = q

        """
        scale = self._scale
        k = self._k
        dps = self._dps
        
        z_approx = scale * chi.ppf(q, k)
        
        epsilon = scale * 0.001
        lb = z_approx - epsilon
        ub = z_approx + epsilon

        f = lambda z: self._cdf_notTruncated(-np.inf, z, dps)

        z = find_root(f, q, lb, ub, tol)

        return z 
        

class truncated_chi2(truncated):

    """

    >>> from selection.constraints.intervals import intervals
    >>> I = intervals.intersection(intervals((-1, 6)),
    ...                            intervals(( 0, 7)),
    ...                           ~intervals((1, 4)))
    >>> distr = truncated_chi2(I, 3, 2.)
    >>> print(abs(distr.cdf(distr.quantile(0.9)) - 0.9) < 0.01)
    True
    """
    def __init__(self, I, k, scale = 1.):
        """
        Create a new object for a truncated_chi distribution

        Parameters
        ----------
        I : intervals
            The intervals the distribution is truncated to

        k : int
            Number of degree of freedom of the distribution

        scale : float
            The distribution is \sim scale * \chi_k

        
        """

        self._k = k
        self._scale = scale
        truncated.__init__(self, I)

    def _cdf_notTruncated(self, a, b, dps):
        """
        Compute the probability of being in the interval (a, b)
        for a variable with a chi distribution (not truncated)
        
        Parameters
        ----------
        a, b : float
            Bounds of the interval. Can be infinite.

        dps : int
            Decimal precision (decimal places). Used in mpmath

        Returns
        -------
        p : float
            The probability of being in the intervals (a, b)
            P( a < X < b)
            for a non truncated variable

        """
        scale = self._scale
        k = self._k

        dps_temp = mp.mp.dps
        mp.mp.dps = dps

        a = max(0, a)
        b = max(0, b)

        cdf = mp.gammainc(1./2 * k, 
                         1./2*(a/scale), 
                         1./2*(b/scale), 
                         regularized=True)
        mp.mp.dps = dps_temp
        return cdf

    def _pdf_notTruncated(self, z, dps):
        scale = self._scale
        k = self._k
        dps = self._dps

        return chi2.pdf(z/scale, k)

    def _quantile_notTruncated(self, q, tol=1.e-6):
        """
        Compute the quantile for the non truncated distribution

        Parameters
        ----------
        q : float
            quantile you want to compute. Between 0 and 1

        tol : float
            precision for the output

        Returns
        -------
        x : float
            x such that P(X < x) = q

        """
        scale = self._scale
        k = self._k
        dps = self._dps
        
        z_approx = scale * chi.ppf(q, k)
        
        epsilon = scale * 0.001
        lb = z_approx - epsilon
        ub = z_approx + epsilon

        f = lambda z: self._cdf_notTruncated(-np.inf, z, dps)

        z = find_root(f, q, lb, ub, tol)

        return z 

    def _pdf_notTruncated(self, z, dps):
        scale = self._scale
        k = self._k
        #dps = self._dps

        return chi2.pdf(z/scale, k)

    def _quantile_notTruncated(self, q, tol=1.e-6):
        """
        Compute the quantile for the non truncated distribution

        Parameters
        ----------
        q : float
            quantile you want to compute. Between 0 and 1

        tol : float
            precision for the output

        Returns
        -------
        x : float
            x such that P(X < x) = q

        """
        scale = self._scale
        k = self._k
        dps = self._dps
        
        z_approx = scale * chi2.ppf(q, k)
        
        epsilon = scale * 0.001
        lb = z_approx - epsilon
        ub = z_approx + epsilon

        f = lambda z: self._cdf_notTruncated(-np.inf, z, dps)

        z = find_root(f, q, lb, ub, tol)

        return z 
