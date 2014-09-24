"""
This module implements the class `truncated_gaussian` which 
performs (conditional) UMPU tests for Gaussians
restricted to a set of intervals.

"""
import numpy as np
from scipy.stats import chi
from mpmath import fsum

import matplotlib
import matplotlib.pyplot as plt

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
    def __init__(self, I):
        """
        Create a new truncated distribution object
        This method is abstract : it has to be overriden

        Parameters
        ----------
        
        I : intervals
            The intervals the distribution is truncated to

        """
        self._I = I

        dps = 15
        not_precise = True
        while not_precise:
            dps *= 2.
            Q = [self._cdf_notTruncated(a, b, dps) for a, b in I]
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

        WARNING : This method is deprecated if not overriden. It can be
        very slow
        """
        
        warnings.warn("""Deprecated to use the general quantile_notTruncated 
        method : it should be overrriden""", DeprecationWarning)

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
            sf(z) = P( X > z | X is in I )
        
        """
        I = self._I
        Q, sumQ = self._Q, self._sumQ
        N = len(Q)
        dps = self._dps

        k, (a, b) = min( (k, (a, b))  for k, (a, b) in enumerate(I) if b > z)

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
            cdf(z) = P( X < z | X is in I )
        
        
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
       
        I = self._I
        dps = self._dps

        if I(z):
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
        I = self._I
        dps = self._dps
        
        cum_sum = np.cumsum( Q )


        k = min( i for i, c in enumerate(cum_sum) if c > q*sumQ )
        a, b = I[k]

        
        q_notTruncated = q*sumQ + self._cdf_notTruncated(-np.inf, a, dps)
        if k>0:
            q_notTruncated -= cum_sum[k-1]

        q_notTruncated = float(q_notTruncated)
        z = self._quantile_notTruncated(q_notTruncated, tol)
        
        return z

    def plt_cdf(self):
        # import matplotlib
        # import matplotlib.pyplot as plt
        print 1
        fig, ax = plt.subplots(1, 1)
        print 2
        l = self.quantile(0.01)
        u = self.quantile(0.99)
        print 2.5
        x = np.linspace(l, u, 100)
        print 3
        cdf_x = [self.cdf(t) for t in x]
        print 4
        ax.plot(x, cdf_x, 'r-', lw=5, alpha=0.6, label='cdf')
        print 5
        plt.show()

    def plt_pdf(self):
        import matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(self.quantile(0.01), self.quantile(0.99), 100)
        pdf_x = [self.pdf(t) for t in x]
        ax.plot(x, pdf_x, 'r-', lw=5, alpha=0.6, label='pdf')
        plt.show()
        


    
    

        
class truncated_old(object):
    
    """
    A distribution, truncated to
    """

    # def __init__(self, 
    #              intervals, 
    #              pdf, 
    #              cdf, 
    #              sf=None, 
    #              quantile=None, 
    #              logcdf=None,
    #              logsf=None):

    #     self.intervals = np.sort(np.array(intervals), 0)

    #     D = self.intervals[:,1]-self.intervals[:,0]
    #     I = self.intervals[D != 0]
    #     self._cutoff_array = I.reshape(-1)

    #     self.pdf_R = pdf
    #     self.cdf_R = cdf

    #     if quantile==None:
    #         quantile = (lambda q: quantile_R(q, cdf))
    #     self.quantile_R = quantile

    #     if sf == None:
    #         sf = lambda x: 1 - cdf(x)
    #     self.sf_R = sf

    #     if logcdf==None:
    #         logcdf = lambda x: np.log(cdf(x))
    #     self.logcdf_R = logcdf
        
    #     if logsf==None:
    #         logsf = lambda x: np.log(sf(x))
    #     self.logsf_R = logsf

    #     self._parameter_changed()

    
    

    # private method to update P and D after a change of parameters

    # def _parameter_changed(self):
    #     I = self.intervals

    #     # cdf_I = self.cdf_R(I)
    #     # self.P = cdf_I[:, 1] - cdf_I[:, 0]
    #     # self.P = (self.P).reshape(-1)
    #     # print "P : ", self.P, "cdf_R", self.cdf_R(I), "intervals", I
        
    #     # self.D = self.pdf_R(I).reshape(-1)

    #     # sf_I = self.sf_R(I)
    #     # self.Q = sf_I[:, 0] - sf_I[:, 1]
    #     # self.Q = (self.Q).reshape(-1)
    #     # print "Q : ", self.Q

    #     dps = 30
    #     not_precise = True
    #     while not_precise:
    #         dps *= 2.
    #         Q = [self.sf_R(i[0], i[1], dps) for i in self.intervals]
    #         not_precise = (fsum(Q) == 0.)

    #     self._dps = dps
    #     self.Q = Q



    #@staticmethod
    # def twosided(thresh, pdf, cdf):
    #     thresh = np.fabs(thresh)
    #     return truncated([(-np.inf,-thresh),(thresh,np.inf)],
    #                               pdf=pdf, cdf=cdf)
    
    
    # def cdf(self, z):
    #     P = self.P
    #     N, = P.shape

    #     # Cette ligne est hideusement laide

    #     k = max(k for k in range(N) if self.intervals[k, 0] < z)
    #     a, b = self.intervals[k]

    #     cdf = P[:k].sum() 
    #     cdf += self.cdf_R(min(b, z)) - self.cdf_R(a)
    #     cdf /= P.sum()
    #     return cdf


    # def sf(self, z):
    #     Q = self.Q
    #     N = len(Q)

    #     k = min(k for k in range(N) if self.intervals[k, 1] > z)
    #     a, b = self.intervals[k]

    #     sf = fsum(Q[k+1:])
    #     sf += self.sf_R(max(a, z), b, self._dps)
    #     sf /= fsum(Q)
            
    #     return sf


    ## Check, because the end of the function was strange for the original
    # def quantile(self, q):
    #     P = self.P
    #     Psum = P.sum()
    #     Csum = np.cumsum(np.array([0]+list(P)))

    #     # Je n'aime pas cette ecriture, 0 n'a rien de canonique 
    #     # dans ce probleme
    #     k = max(np.nonzero(Csum < Psum*q)[0])


    #     pnorm_increment = Psum*q - Csum[k]
        
    #     return self.quantile_R(self.cdf_R(self.intervals[k][0]) + pnorm_increment)
        
    
        
    ## OK
    def right_endpoint(self, left_endpoint, alpha):
        alpha1 = self.cdf(left_endpoint)
        if (alpha1 > alpha):
            return np.nan
        alpha2 = np.array(alpha - alpha1)
        return self.quantile(1-alpha2)

    def plt_cdf(self):
        
        print "imported"
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(self.quantile(0.01), self.quantile(0.99), 100)
        print"lin space : ok"
        cdf_x = [self.cdf(t) for t in x]
        print "values : ok"
        ax.plot(x, cdf_x, 'r-', lw=5, alpha=0.6, label='chi pdf')
        plt.show()
 

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
    max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )

    # bisect (slow but sure) until solution is obtained
    for _ in xrange(max_iter):
        c, fc  = (a+b)/2, f((a+b)/2)
        if fc > y: a = c
        elif fc < y: b = c
    
    return c




# def quantile_R(q, cdf, tol=1e-6):
#     l, u = -np.inf, np.inf

#     # First step : we find some upper and lower bounds
#     t = 0.
#     if cdf(t) <= q:
#         l = t
#         t = 1.
#         while cdf(t) < q:
#             t *= 2
#         u = t
#     else:
#         u = t
#         t = -1
#         while cdf(t) > q:
#             t *= 2
#         l = t

#     while np.fabs(u - l) > tol:
#         t = (l+u)/2
#         if cdf(t) < q:
#             l = t
#         else:
#             u = t

#     return (l+u)/2
        

