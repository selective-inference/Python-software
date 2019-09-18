import numpy as np
from mpmath import mp
from scipy.stats import t as tdist
from .base import truncated
from .F import sf_F

# quantile = tdist.ppf
# 

def sf_T(df):

    def sf(a, b=np.inf, dps=15):
        dps_temp = mp.dps
        mp.dps = dps

        # case1: sign(a) == sign(b)
        
        d1 = 1.
        d2 = df
        scale = 1.

        if a*b >= 0.:
            a, b = sorted([a**2, b**2])
            tmp_a = d1*a/d2
            beta_a = tmp_a / (1. + tmp_a)
            if b < np.inf:
                tmp_b = d1*b/d2
                beta_b = tmp_b / (1. + tmp_b)
            else:
                beta_b = 1.
            sf = mp.betainc(d1/2., d2/2., 
                            x1=beta_a, x2=beta_b,
                            regularized=True)
        # case2: different signs

        else:
            a, b = [a**2, b**2]
            if a < np.inf:
                tmp_a = d1*a/d2
                beta_a = tmp_a / (1. + tmp_a)
            else:
                beta_a = 1.
            if b < np.inf:
                tmp_b = d1*b/d2
                beta_b = tmp_b / (1. + tmp_b)
            else:
                beta_b = 1.
            sf = (mp.betainc(d1/2., d2/2., 
                            x1=0, x2=beta_a, 
                            regularized=True) + 
                  mp.betainc(d1/2., d2/2., 
                            x1=0, x2=beta_b, 
                            regularized=True)) 
        mp.dps = dps_temp
        return sf / 2.

    return sf

class truncated_T(truncated):

    dps = 20

    def __init__(self, intervals, df):

        self._T = tdist(df)
        self._Tsf = sf_T(df)

        truncated.__init__(self, intervals)

    def _cdf_notTruncated(self, a, b, dps):
        """
        Compute the probability of being in the interval (a, b)
        for a variable with a T distribution (not truncated)
        
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

        return self._Tsf(a, b, dps=self.dps)

    def _pdf_notTruncated(self, z, dps):
        """
        Compute the density for the non truncated T distribution

        Parameters
        ----------
        z : float
            Value where density is to be calculated.

        Returns
        -------
        d : float
            pdf at z

        """

        return self._T.pdf(z)

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

        return self._T.ppf(q)
