"""
This module implements the class `truncated_gaussian` which 
performs (conditional) UMPU tests for Gaussians
restricted to a set of intervals.

"""
import numpy as np
from scipy.stats import chi
from mpmath import *






class truncated(object):
    
    """
    A distribution, truncated to
    """
    def __init__(self, 
                 intervals, 
                 pdf, 
                 cdf, 
                 sf=None, 
                 quantile=None, 
                 logcdf=None,
                 logsf=None):


        self.intervals = np.array(intervals)

        D = self.intervals[:,1]-self.intervals[:,0]
        I = self.intervals[D != 0]
        self._cutoff_array = I.reshape(-1)

        self.pdf_R = pdf
        self.cdf_R = cdf

        if quantile==None:
            quantile = (lambda q: quantile_R(q, cdf))
        self.quantile_R = quantile

        if sf == None:
            sf = lambda x: 1- cdf(x)
        self.sf_R = sf

        if logcdf==None:
            logcdf = lambda x: np.log(cdf(x))
        self.logcdf_R = logcdf
        
        if logsf==None:
            logsf = lambda x: np.log(sf(x))
        self.logsf_R = logsf

        self._parameter_changed()

    def __array__(self):
        return self.intervals
    
    # @property
    # def negated(self):
    #     if not hasattr(self,"_negated"):
    #         pdf_R = lambda x: self.pdf_R(-x)
    #         cdf_R = lambda x: 1. - self.cdf_R(-x)
    #         quantile_R = lambda q: - self.quantile_R(1-q)
    #         intervals = np.asarray(-self._cutoff_array[::-1])
    #         self._negated = truncated(intervals, pdf_R, cdf_R, quantile_R)
    #     return self._negated
    

    # private method to update P and D after a change of parameters

    def _parameter_changed(self):
        I = self.intervals

        # cdf_I = self.cdf_R(I)
        # self.P = cdf_I[:, 1] - cdf_I[:, 0]
        # self.P = (self.P).reshape(-1)
        # print "P : ", self.P, "cdf_R", self.cdf_R(I), "intervals", I
        
        # self.D = self.pdf_R(I).reshape(-1)

        # sf_I = self.sf_R(I)
        # self.Q = sf_I[:, 0] - sf_I[:, 1]
        # self.Q = (self.Q).reshape(-1)
        # print "Q : ", self.Q

        dps = 30
        not_precise = True
        while not_precise:
            dps *= 2.
            Q = [self.sf_R(i[0], i[1], dps) for i in self.intervals]
            not_precise = (fsum(Q) == 0.)

        self._dps = dps
        self.Q = Q


        



    #@staticmethod
    # def twosided(thresh, pdf, cdf):
    #     thresh = np.fabs(thresh)
    #     return truncated([(-np.inf,-thresh),(thresh,np.inf)],
    #                               pdf=pdf, cdf=cdf)
    
    
    def cdf(self, z):
        P = self.P
        N, = P.shape
        # Cette ligne est hideusement laide

        k = max(k for k in range(N) if self.intervals[k, 0] < z)
        a, b = self.intervals[k]

        cdf = P[:k].sum() 
        cdf += self.cdf_R(min(b, z)) - self.cdf_R(a)
        cdf /= P.sum()
        return cdf


    def sf(self, z):
        Q = self.Q
        N =len(Q)

        k = min(k for k in range(N) if self.intervals[k, 1] > z)
        a, b = self.intervals[k]

        sf = fsum(Q[k+1:])
        sf += self.sf_R(max(a, z), b, self._dps)
        sf /= fsum(Q)
            
        return sf


    ## Check, because the end of the function was strange for the original
    def quantile(self, q):
        P = self.P
        Psum = P.sum()
        Csum = np.cumsum(np.array([0]+list(P)))

        # Je n'aime pas cette ecriture, 0 n'a rien de canonique 
        # dans ce probleme
        k = max(np.nonzero(Csum < Psum*q)[0])


        pnorm_increment = Psum*q - Csum[k]
        
        return self.quantile_R(self.cdf_R(self.intervals[k][0]) + pnorm_increment)
        
    
        
    ## OK
    def right_endpoint(self, left_endpoint, alpha):
        alpha1 = self.cdf(left_endpoint)
        if (alpha1 > alpha):
            return np.nan
        alpha2 = np.array(alpha - alpha1)
        return self.quantile(1-alpha2)

    def plt_cdf(self):
        import matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(self.quantile(0.01), self.quantile(0.99), 100)
        cdf_x = [self.cdf(t) for t in x]
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




def quantile_R(q, cdf, tol=1e-6):
    l, u = -np.inf, np.inf

    # First step : we find some upper and lower bounds
    t = 0.
    if cdf(t) <= q:
        l = t
        t = 1.
        while cdf(t) < q:
            t *= 2
        u = t
    else:
        u = t
        t = -1
        while cdf(t) > q:
            t *= 2
        l = t

    while np.fabs(u - l) > tol:
        t = (l+u)/2
        if cdf(t) < q:
            l = t
        else:
            u = t

    return (l+u)/2
        

def pdf_chi(k, scale):
    return lambda x: chi.pdf(x/scale, k)

def cdf_chi(k, scale):
    return lambda x: chi.cdf(x/scale, k)

def sf_chi(k, scale):
    return lambda x: chi.sf(x/scale, k)

def quantile_chi(k, scale):
    return lambda q: scale * chi.ppf(q, k)

        
    
# def trunc_chi_distr(intervals, k, scale=1.):
#     tr = truncated(\
#                    intervals, 
#                    pdf_chi(k, scale), 
#                    cdf_chi(k, scale), 
#                    quantile_chi(k, scale))
#     return tr


def test_truncated():
    intervals = [[0., 1.], [3., 4.]]
    tr = truncated(intervals, pdf_chi(3, 2.), cdf_chi(3, 2.), quantile_chi(3, 2.))
    tr.plt_cdf()
                   
