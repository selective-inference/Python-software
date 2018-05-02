from rpy2.robjects.packages import importr
from rpy2 import robjects

SLOPE = importr('SLOPE')

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from selection.tests.instance import gaussian_instance

import numpy as np
from regreg.atoms.slope import slope
import regreg.api as rr

from selection.randomized.slope import slope
from statsmodels.distributions import ECDF

import matplotlib.pyplot as plt

def test_slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian", sigma = None):
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

    return np.asarray(result.rx2('beta')), np.asarray(result.rx2('E')), \
           np.asarray(result.rx2('lambda_seq')), np.asscalar(np.array(result.rx2('sigma')))

def compare_outputs_SLOPE_weights(n=500, p=100, signal_fac=1., s=5, sigma=3., rho=0.35):

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
    r_beta, r_E, r_lambda_seq, r_sigma = test_slope_R(X,
                                                      Y,
                                                      W = None,
                                                      normalize = True,
                                                      choice_weights = "gaussian",
                                                      sigma = sigma_)
    print("estimated sigma", sigma_, r_sigma)
    print("weights output by R", r_lambda_seq)
    print("output of est coefs R", r_beta)

    pen = slope(r_sigma * r_lambda_seq, lagrange=1.)

    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve()
    print("output of est coefs python", soln)

    print("relative difference in solns", np.linalg.norm(soln-r_beta)/np.linalg.norm(r_beta))

# #compare_outputs_SLOPE_weights()

# def test0_randomized_slope(n=500, p=100, signal_fac=1., s=5, sigma=3., rho=0.35,
#                      randomizer_scale= np.sqrt(0.25),
#                      solve_args={'tol':1.e-12, 'min_its':50}):

#     inst = gaussian_instance
#     signal = np.sqrt(signal_fac * 2. * np.log(p))
#     X, Y, beta = inst(n=n,
#                       p=p,
#                       signal=signal,
#                       s=s,
#                       equicorrelated=False,
#                       rho=rho,
#                       sigma=sigma,
#                       random_signs=True)[:3]

#     sigma_ = np.sqrt(np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p))
#     r_beta, r_E, r_lambda_seq, r_sigma = test_slope_R(X,
#                                                       Y,
#                                                       W=None,
#                                                       normalize=True,
#                                                       choice_weights="gaussian",
#                                                       sigma=sigma_)

#     pen = slope(r_sigma * r_lambda_seq, lagrange=1.)

#     loglike = rr.glm.gaussian(X, Y, coef=1., quadratic=None)
#     _initial_omega = randomizer_scale * sigma_* np.random.standard_normal(p)
#     quad = rr.identity_quadratic(0, 0, -_initial_omega, 0)
#     problem = rr.simple_problem(loglike, pen)
#     initial_soln = problem.solve(quad, **solve_args)
#     initial_subgrad = -(loglike.smooth_objective(initial_soln, 'grad') + quad.objective(initial_soln, 'grad'))

#     indices = np.argsort(-np.abs(initial_soln))
#     sorted_soln = initial_soln[indices]

#     cur_indx_array = []
#     cur_indx_array.append(0)
#     cur_indx = 0
#     pointer = 0
#     signs_cluster = []
#     for j in range(p-1):
#         if np.abs(sorted_soln[j+1]) != np.abs(sorted_soln[cur_indx]):
#             cur_indx_array.append(j+1)
#             cur_indx = j+1
#             sign_vec = np.zeros(p)
#             sign_vec[np.arange(j+1-cur_indx_array[pointer]) + cur_indx_array[pointer]] = \
#                 np.sign(initial_soln[indices[np.arange(j+1-cur_indx_array[pointer]) + cur_indx_array[pointer]]])
#             signs_cluster.append(sign_vec)
#             pointer = pointer + 1
#             if sorted_soln[j+1]== 0:
#                 break

#     signs_cluster = np.asarray(signs_cluster).T
#     X_clustered = X[:, indices].dot(signs_cluster)
#     print("start indices of clusters", indices, cur_indx_array, signs_cluster.shape, X_clustered.shape)

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
        r_beta, r_E, r_lambda_seq, r_sigma = test_slope_R(X,
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
            if target == "selected":
                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            else:
                beta_target = beta[nonzero]
            if use_MLE:
                estimate, _, _, pval, intervals, _ = conv.selective_MLE(target=target, dispersion=sigma_)
            else:
                _, pval, intervals = conv.summary(target="selected", dispersion=sigma_, compute_intervals=True)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            break

    if True:
        #print(beta_target)
        return pval[beta_target == 0], pval[beta_target != 0], coverage, intervals

def main(nsim=100):

    P0, PA, cover, length_int = [], [], [], []
    
    for i in range(nsim):
        p0, pA, cover_, intervals = test_randomized_slope()

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print('coverage', np.mean(cover))

        # if i % 3 == 0 and i > 0:
        #     U = np.linspace(0, 1, 101)
        #     plt.clf()
        #     if len(P0) > 0:
        #         plt.plot(U, ECDF(P0)(U))
        #     if len(PA) > 0:
        #         plt.plot(U, ECDF(PA)(U), 'r')
        #     plt.plot([0, 1], [0, 1], 'k--')
        #     plt.draw()

main()


