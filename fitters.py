import numpy as np
from scipy.stats import norm as ndist
import rpy2.robjects as rpy
import rpy2.robjects.numpy2ri
rpy.r('library(splines)')

def logit_fit(T, Y, df=20):
    rpy2.robjects.numpy2ri.activate()
    rpy.r.assign('T', T)
    rpy.r.assign('df', df)
    rpy.r.assign('Y', Y.astype(np.int))
    cmd = '''
    Y = as.numeric(Y)
    T = as.matrix(T)
    colnames(T) = c(%s)
    cur_data = data.frame(Y, T)
    M = glm(Y ~ %s, family=binomial(link='logit'), data=cur_data)
    fitfn = function(t) {
        t = data.frame(t)
        colnames(t) = c(%s)
        predict(M, newdata=t, type='link') 
    } 
    ''' % (', '.join(['"T%d"' % i for i in range(1, T.shape[1]+1)]),
           ' + '.join(['ns(T%d, df)' % i for i in range(1, T.shape[1]+1)]),
           ', '.join(['"T%d"' % i for i in range(1, T.shape[1]+1)]))
    rpy.r(cmd)
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

def probit_fit(T, Y, df=20):
    rpy2.robjects.numpy2ri.activate()
    rpy.r.assign('T', T)
    rpy.r.assign('df', df)
    rpy.r.assign('Y', Y.astype(np.int))
    cmd = '''
    Y = as.numeric(Y)
    T = as.matrix(T)
    colnames(T) = c(%s)
    cur_data = data.frame(Y, T)
    M = glm(Y ~ %s, family=binomial(link='probit'), data=cur_data)
    fitfn = function(t) {
        t = data.frame(t)
        colnames(t) = c(%s)
        predict(M, newdata=t, type='link') 
    } 
    ''' % (', '.join(['"T%d"' % i for i in range(1, T.shape[1]+1)]),
           ' + '.join(['ns(T%d, df)' % i for i in range(1, T.shape[1]+1)]),
           ', '.join(['"T%d"' % i for i in range(1, T.shape[1]+1)]))
    rpy.r(cmd)
    rpy2.robjects.numpy2ri.deactivate()

    # this is a little fragile obviously as someone might overwrite fitfn

    def fitfn(t):
        rpy2.robjects.numpy2ri.activate()
        fitfn_r = rpy.r('fitfn')
        val = np.asarray(fitfn_r(t))
        rpy2.robjects.numpy2ri.deactivate()
        return ndist.cdf(val)

    return fitfn
