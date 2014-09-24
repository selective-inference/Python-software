import numpy as np
import mpmath as mp

from .base import truncated


def sf_F(d1, d2, scale):

    def sf(a, b=np.inf, dps=15):
        dps_temp = mp.mp.dps
        mp.mp.dps = dps

        tmp_a = d1*a/d2
        tmp_b = d1*b/d2
        beta_a = tmp_a / (1. + tmp_a)
        beta_b = tmp_b / (1. + tmp_b)
        if b == np.inf:
            beta_b = 1.
        sf = mp.betainc(d1/2., d2/2., 
                        x1=beta_a, x2=beta_b,
                        regularized=True)
        mp.mp.dps = dps_temp
        return sf

    return sf

def null_f(x):
    raise ValueError("Shouldn't be called")
    return 0


class truncated_F(truncated):
    def __init__(self, intervals, d1, d2, scale=1):
        self._d1 = d1
        self._d2 = d2
        self._scale = scale

        truncated.__init__(self,
                           intervals,
                           null_f,
                           null_f,
                           sf_F(d1, d2, scale),
                           null_f)
