"""
This module implements the class `truncated_gaussian` which 
performs (conditional) UMPU tests for Gaussians
restricted to a set of intervals.

"""
import numpy as np
from scipy.stats import chi
from mpmath import fsum

import warnings

from abc import ABCMeta, abstractmethod


class truncated(object):
    """
    A distribution, truncated to a union of intervals

    HOW TO MAKE A SUBCLASS : 
    You have to implement : 

    __init__(self, args*) : It has to call the method from the base class
        Since the method is abstract, you can't have an instance of the
        subclass if the method __init__ is not implemented
    
    _cdf_notTruncated(self, a, b, dps) :
    
    With these two methods, you can use : 
        -> cdf
        -> sf

    You should implement : 

    _pdf_notTruncated(self, z, dps) : it allows you to use : 
        -> pdf
        -> plt_pdf (if you also have _quantile_notTruncated)

    _quantile_notTruncated(self, q, tol) : it allows  you to use : 
        -> quantile
        -> rvs 
        -> plt_cdf
        -> plt_pdf (if you also have  _pdf_notTruncated)

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, intervals):
        """
        Create a new truncated distribution object
        This method is abstract : it has to be overriden

        Parameters
        ----------
        
        intervals : [(float, float)]
            The intervals the distribution is truncated to

        """
        self.intervals = intervals

        dps = 15
        not_precise = True
        while not_precise:
            Q = [self._cdf_notTruncated(a, b, dps) for a, b in intervals]
            dps *= 2
            not_precise = (fsum(Q) == 0.)

        self._sumQ = fsum(Q)
        self._dps = dps
        self._Q = Q

               
    @abstractmethod
    def _cdf_notTruncated(self, a, b, dps=15):
        """
        Compute the probability of being in the interval (a, b)
        for a variable with this distribution (not truncated)
        
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

        WARNING : This is the fundamental method of the truncated class
        It has to be overriden for each distribution
        """
        pass



    def _quantile_notTruncated(self, q, dps, tol=1.e-6):
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

        WARNING : This method is deprecated if not overriden. It can be
        very slow
        """
        
        warnings.warn("""Deprecated to use the general quantile_notTruncated 
        method : it should be overrriden""", DeprecationWarning)

        if dps is None:
            dps = self._dps
        f = lambda x: cdf_notTruncated(-np.inf, x, dps)

        lb, ub = -1.e8, 1.e8
        x = find_root(f, y, lb, ub, tol=1e-6)

        return x



    def rvs(self, size=1):
        """
        Sample a random variable from the truncated disribution

        Parameters
        ----------
        size : int
           Number of samples. Default : 1

        Returns
        -------
        X : np.array
            array of sample

        """
        if not hasattr(self, '_quantile_notTruncated'):
            raise NotImplementedError( \
                """The 'quantile_notTruncated' method 
                should be implemented in order to use the truncated 
                rvs method"""
            )

        U = np.random.uniform(size=size)
        X = np.array([self.quantile(u) for u in U])
        return X


    def sf(self, z):
        """
        Compute the survival function of the truncated distribution

        Parameters
        ----------
        z : float
            Minimum bound of the interval

        Returns
        -------
        sf : float
            The survival function of the truncated distribution
            sf(z) = P( X > z | X is in intervals )
        
        """
        intervals = self.intervals
        Q, sumQ = self._Q, self._sumQ
        N = len(Q)
        dps = self._dps

        k, (a, b) = min( (k, (a, b))  for k, (a, b) in enumerate(intervals) if b > z)

        sf = fsum(Q[k+1:]) + self._cdf_notTruncated(max(a, z), b, dps)
        sf /= sumQ
            
        return sf


    def cdf(self, z):
        """
        Compute the survival function of the truncated distribution

        Parameters
        ----------
        z : float
            Minimum bound of the interval

        Returns
        -------
        cdf : float
            function  The cumulative distribution function of the 
            truncated distribution
            cdf(z) = P( X < z | X is in intervals )
        
        
        WARNING : This method only use the sf method : it is never going to be 
        more precise
        """
        cdf = 1. - self.sf(z)
        return cdf


    def pdf(self, z):
        """
        Compute the probability distribution funtion of the
        truncated distribution

        Parameters
        ----------
        z : float
            
        Returns
        -------
        p : float
            p(z) such that E[f(X)] = \int f(z)p(z)dz

        """
        
        if not hasattr(self, '_pdf_notTruncated'):
            raise NotImplementedError( \
                """The 'pdf_notTruncated' 
                should be implemented in order to use the truncated 
                pdf method"""
            )
       
        intervals = self.intervals
        dps = self._dps

        if intervals(z):
            p = self._pdf_notTruncated(z, dps)
            p /= self._sumQ
        else:
            p = 0

        return p

    def quantile(self, q, tol=1.e-6):
        if not hasattr(self, '_quantile_notTruncated'):
            raise NotImplementedError( \
                """The 'quantile_notTruncated' method 
                should be implemented in order to use the truncated 
                quantile method"""
            )


        Q = self._Q
        sumQ = self._sumQ
        intervals = self.intervals
        dps = self._dps
        
        cum_sum = np.cumsum( Q )


        k = min( i for i, c in enumerate(cum_sum) if c > q*sumQ )
        a, b = intervals[k]

        
        q_notTruncated = q*sumQ + self._cdf_notTruncated(-np.inf, a, dps)
        if k>0:
            q_notTruncated -= cum_sum[k-1]

        q_notTruncated = float(q_notTruncated)
        z = self._quantile_notTruncated(q_notTruncated, tol)
        
        return z

def find_root(f, y, lb, ub, tol=1e-6):
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
    try:
        max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )
    except OverflowError:
        warnings.warn('root finding failed, returning np.nan')
        return np.nan
        

    # bisect (slow but sure) until solution is obtained
    for _ in range(max_iter):
        try:
            c, fc  = (a+b)/2, f((a+b)/2)
            if fc > y: a = c
            elif fc < y: b = c
        except OverflowError:
            warnings.warn('root finding failed, returning np.nan')
            return np.nan

    return c
        

