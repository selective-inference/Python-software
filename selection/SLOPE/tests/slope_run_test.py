
from rpy2.robjects.packages import importr
from rpy2 import robjects

SLOPE = importr('SLOPE')

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np
import sys

from regreg.atoms.slope import slope

import regreg.api as rr


def test_slope_R(X, Y, W):
    robjects.r('''
    slope = function(X, Y, W=NA, fdr = 0.1){

      if(is.na(W))
      {
        result = SLOPE(X, Y, fdr = fdr, lambda = "gaussian", sigma =1)
       } else{
        result = SLOPE(X, Y, fdr = fdr, lambda = W, sigma =1)
      }

      return(list(beta = result$beta, E = result$selected))
    }''')

    r_slope = robjects.globalenv['slope']

    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_Y = robjects.r.matrix(Y, nrow=n, ncol=1)
    r_W = robjects.r.matrix(W, nrow=p, ncol=1)
    result = r_slope(r_X, r_Y, r_W)

    return result[0]

def compare_outputs():

    n, p = 500, 200

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    W = np.linspace(3, 3.5, p)[::-1]

    output_R = test_slope_R(X, Y, W)

    pen = slope(W, lagrange=1.)
    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve()

compare_outputs()
