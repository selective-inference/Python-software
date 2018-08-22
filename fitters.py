import numpy as np
from scipy.stats import norm as ndist
import rpy2.robjects as rpy
import rpy2.robjects.numpy2ri
rpy.r('library(splines)')

def logit_fit(T, Y):
    rpy2.robjects.numpy2ri.activate()
    rpy.r.assign('T', T)
    rpy.r.assign('Y', Y.astype(np.int))
    rpy.r('''
    Y = as.numeric(Y)
    T = as.numeric(T)
    M = glm(Y ~ ns(T, 10), family=binomial(link='logit'))
    fitfn = function(t) { predict(M, newdata=data.frame(T=t), type='link') } 
    ''')
    rpy2.robjects.numpy2ri.deactivate()

    # this is a little fragile obviously as someone might overwrite fitfn

    def fitfn(t):
        rpy2.robjects.numpy2ri.activate()
        fitfn_r = rpy.r('fitfn')
        val = np.asarray(fitfn_r(t))
        rpy2.robjects.numpy2ri.deactivate()
        result = np.zeros(t.shape)
        test = val < 0
        result[test] = np.exp(val[test]) / (1 + np.exp(val[test]))
        result[~test] = 1 / (1 + np.exp(-val[~test]))
        return result

    return fitfn

def probit_fit(T, Y):
    rpy2.robjects.numpy2ri.activate()
    rpy.r.assign('T', T)
    rpy.r.assign('Y', Y.astype(np.int))
    rpy.r('''
    Y = as.numeric(Y)
    T = as.numeric(T)
    M = glm(Y ~ ns(T, 10), family=binomial(link='probit'))
    fitfn = function(t) { predict(M, newdata=data.frame(T=t), type='link') } 
    ''')
    rpy2.robjects.numpy2ri.deactivate()

    # this is a little fragile obviously as someone might overwrite fitfn

    def fitfn(t):
        rpy2.robjects.numpy2ri.activate()
        fitfn_r = rpy.r('fitfn')
        val = np.asarray(fitfn_r(t))
        rpy2.robjects.numpy2ri.deactivate()
        return ndist.cdf(val)

    return fitfn
