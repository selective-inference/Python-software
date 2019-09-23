
from ...tests.instance import gaussian_instance

import numpy as np
from regreg.atoms.slope import slope as slope_atom
import regreg.api as rr

from ..slope import slope
from ..lasso import full_targets
from ...tests.decorators import rpy_test_safe

try:
    from rpy2.robjects.packages import importr
    from rpy2 import robjects
    import rpy2.robjects.numpy2ri
    rpy_loaded = True
except ImportError:
    rpy_loaded = False

if rpy_loaded:
    def slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian", sigma = None):
        rpy2.robjects.numpy2ri.activate()
        robjects.r('''
        slope = function(X, Y, W , normalize, choice_weights, sigma, fdr = NA){
          if(is.na(sigma)){
          sigma=NULL} else{
          sigma = as.matrix(sigma)[1,1]}
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
            result = SLOPE(X, Y, fdr = fdr, lambda = lambda, normalize = normalize, sigma = sigma)
           } else{
            result = SLOPE(X, Y, fdr = fdr, lambda = W, normalize = normalize, sigma = sigma)
          }
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

        if sigma is None:
            r_sigma = robjects.NA_Logical
        else:
            r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)

        result = r_slope(r_X, r_Y, r_W, r_normalize, r_choice_weights, r_sigma)

        result = np.asarray(result.rx2('beta')), np.asarray(result.rx2('E')), \
            np.asarray(result.rx2('lambda_seq')), np.asscalar(np.array(result.rx2('sigma')))
        rpy2.robjects.numpy2ri.deactivate()

        return result

@rpy_test_safe(libraries=['SLOPE'])
def test_outputs_SLOPE_weights(n=500, p=100, signal_fac=1., s=5, sigma=3., rho=0.35):

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

    sigma_ = np.sqrt(np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p))

    r_beta, r_E, r_lambda_seq, r_sigma = slope_R(X,
                                                 Y,
                                                 W = None,
                                                 normalize = True,
                                                 choice_weights = "gaussian",
                                                 sigma = sigma_)
    print("estimated sigma", sigma_, r_sigma)
    print("weights output by R", r_lambda_seq)
    print("output of est coefs R", r_beta)

    pen = slope_atom(r_sigma * r_lambda_seq, lagrange=1.)

    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve()
    print("output of est coefs python", soln)

    print("relative difference in solns", np.linalg.norm(soln-r_beta)/np.linalg.norm(r_beta))

@rpy_test_safe(libraries=['SLOPE'])
def test_randomized_slope(n=500, p=100, signal_fac=1.2, s=5, sigma=1., rho=0.35, randomizer_scale= np.sqrt(0.25),
                          target = "full", use_MLE=True):

    while True:
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

        sigma_ = np.sqrt(np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p))
        r_beta, r_E, r_lambda_seq, r_sigma = slope_R(X,
                                                     Y,
                                                     W=None,
                                                     normalize=True,
                                                     choice_weights="gaussian", #put gaussian
                                                     sigma=sigma_)

        conv = slope.gaussian(X,
                              Y,
                              r_sigma * r_lambda_seq,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            if target == 'full':
                (observed_target, 
                 cov_target, 
                 cov_target_score, 
                 alternatives) = full_targets(conv.loglike, 
                                              conv._W, 
                                              nonzero, dispersion=sigma_)
            elif target == 'selected':
                (observed_target, 
                 cov_target, 
                 cov_target_score, 
                 alternatives) = selected_targets(conv.loglike, 
                                                  conv._W, 
                                                  nonzero, dispersion=sigma_)

            if target == "selected":
                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            else:
                beta_target = beta[nonzero]
            if use_MLE:

                estimate, _, _, pval, intervals, _ = conv.selective_MLE(observed_target, 
                                                                        cov_target, 
                                                                        cov_target_score)
            else:
                _, pval, intervals = conv.summary(observed_target, 
                                                  cov_target, 
                                                  cov_target_score, 
                                                  alternatives, 
                                                  compute_intervals=True)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            break

    if True:
        return pval[beta_target == 0], pval[beta_target != 0], coverage, intervals

def main(nsim=100):

    P0, PA, cover, length_int = [], [], [], []
    
    for i in range(nsim):
        p0, pA, cover_, intervals = test_randomized_slope()

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print('coverage', np.mean(cover))





