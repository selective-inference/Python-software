import numpy as np
from mpmath import mp
from scipy.stats import t as tdist
from .base import truncated
from .F import sf_F

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
    def __init__(self, intervals, df):

        self._T = tdist(df)
        
        truncated.__init__(self,
                           intervals,
                           self._T.pdf,
                           self._T.cdf,
                           sf_T(df),
                           self._T.ppf,
                           self._T.logcdf,
                           self._T.logsf)
