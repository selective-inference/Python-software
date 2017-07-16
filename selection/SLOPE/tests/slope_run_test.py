
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
    slope = function(X, Y, W=NA, fdr = NA, sigma = 1){

      if(is.na(sigma)){
      sigma = NULL}

      if(is.na(fdr)){
      fdr = 0.1 }

      if(is.na(W))
      {
        result = SLOPE(X, Y, fdr = fdr, lambda = "gaussian", sigma = sigma)
       } else{
        result = SLOPE(X, Y, fdr = fdr, lambda = W, sigma = sigma, normalize = FALSE)
      }

      return(list(beta = result$beta, E = result$selected, lambda_seq = result$lambda, sigma = result$sigma))
    }''')

    r_slope = robjects.globalenv['slope']

    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_Y = robjects.r.matrix(Y, nrow=n, ncol=1)
    r_W = robjects.r.matrix(W, nrow=p, ncol=1)
    result = r_slope(r_X, r_Y, r_W)

    return result[0], result[1], result[2], result[3]

def compare_outputs():

    n, p = 500, 50

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    W = np.linspace(3, 3.5, p)[::-1]

    output_R = test_slope_R(X, Y, W)
    r_beta = output_R[0]
    r_E = output_R[1]
    r_lambda_seq = output_R[2]
    r_sigma = output_R[3]
    print("output os est coefs R", r_beta)

    pen = slope(W, lagrange=1.)
    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve()
    print("output os est coefs python", soln)

    print("difference in solns", soln-r_beta)

compare_outputs()
