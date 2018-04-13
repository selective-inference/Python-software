from rpy2.robjects.packages import importr
from rpy2 import robjects

SLOPE = importr('SLOPE')

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from selection.tests.instance import gaussian_instance

import numpy as np
from selection.SLOPE.slope import slope
import regreg.api as rr

def test_slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian"):
    robjects.r('''
    slope = function(X, Y, W=NA, normalize, choice_weights, fdr = NA){
      if(is.na(fdr)){
      fdr = 0.1 }
      if(normalize=="TRUE"){
       normalize = TRUE} else{
       normalize = FALSE}
      if(is.na(W))
      {
        if(choice_weights == "gaussian"){
        lambda = "gaussian"} else{
        lambda = "bhq"}
        result = SLOPE(X, Y, fdr = fdr, lambda = lambda, normalize = normalize)
       } else{
        result = SLOPE(X, Y, fdr = fdr, lambda = W, normalize = normalize)
      }
      print(paste("estimated sigma", class(result$sigma)))
      return(list(beta = result$beta, E = result$selected, lambda_seq = result$lambda, sigma = result$sigma))
    }''')

    r_slope = robjects.globalenv['slope']

    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_Y = robjects.r.matrix(Y, nrow=n, ncol=1)

    if normalize is True:
        r_normalize = robjects.StrVector('True')
    else:
        r_normalize = robjects.StrVector('False')

    if W is None:
        r_W = robjects.NA_Logical
        if choice_weights is "gaussian":
            r_choice_weights  = robjects.StrVector('gaussian')
        elif choice_weights is "bhq":
            r_choice_weights = robjects.StrVector('bhq')

    else:
        r_W = robjects.r.matrix(W, nrow=p, ncol=1)

    result = r_slope(r_X, r_Y, r_W, r_normalize, r_choice_weights)

    return np.asarray(result.rx2('beta')), np.asarray(result.rx2('E')), \
           np.asarray(result.rx2('lambda_seq')), np.asscalar(np.array(result.rx2('sigma')))

def compare_outputs_prechosen_weights():

    n, p = 500, 50

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    W = np.linspace(3, 3.5, p)[::-1]

    output_R = test_slope_R(X, Y, W)
    print("output R", output_R)
    beta_R = output_R[0]
    print("output of est coefs R", beta_R)

    pen = slope(W, lagrange=1.)
    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve()
    print("output of est coefs python", soln)

    print("relative difference in solns", np.linalg.norm(soln-beta_R)/np.linalg.norm(beta_R))

#compare_outputs_prechosen_weights()

# def compare_outputs_SLOPE_weights():
#
#     n, p = 500, 50
#
#     X = np.random.standard_normal((n, p))
#     X -= X.mean(0)[None, :]
#     X /= (X.std(0)[None, :] * np.sqrt(n))
#     beta = np.zeros(p)
#     beta[:5] = 5.
#
#     Y = X.dot(beta) + np.random.standard_normal(n)
#
#     output_R = test_slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian")
#     r_beta = output_R[0]
#     r_lambda_seq = output_R[2]
#     print("output of est coefs R", r_beta)
#
#     W = r_lambda_seq
#     pen = slope(W, lagrange=1.)
#
#     loss = rr.squared_error(X, Y)
#     problem = rr.simple_problem(loss, pen)
#     soln = problem.solve()
#     print("output of est coefs python", soln)
#
#     print("relative difference in solns", np.linalg.norm(soln-r_beta)/np.linalg.norm(r_beta))

def compare_outputs_SLOPE_weights(n=500, p=100, signal_fac=1.1, s=5, sigma=3., rho=0.):

    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      s=s,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]

    r_beta, r_E, r_lambda_seq, r_sigma = test_slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian")
    print("estimated sigma", r_sigma)
    print("output of est coefs R", r_beta)

    pen = slope(r_sigma* r_lambda_seq, lagrange=1.)

    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve()
    print("output of est coefs python", soln)

    print("relative difference in solns", np.linalg.norm(soln-r_beta)/np.linalg.norm(r_beta))

compare_outputs_SLOPE_weights()