import numpy as np
import mpmath as mp

from truncated_bis import truncated


def sf_F(d1, d2, scale):

    def sf(a, b=1., dps=15):
        dps_temp = mp.mp.dps
        mp.mp.dps = dps
        sf = mp.betainc(float(d1)/2, float(d2/2), 
                        x1=(d1*(a/scale))/(d1*(b/scale) + d2), x2=1, 
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
