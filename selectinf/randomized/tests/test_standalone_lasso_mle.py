from __future__ import division, print_function

import numpy as np
import nose.tools as nt

import regreg.api as rr

from selectinf.randomized.lasso import split_lasso, selected_targets
from selectinf.randomized.query import selective_MLE
from selectinf.randomized.approx_reference import approximate_grid_inference

def test_standalone_inference(n=2000, 
                              p=100, 
                              signal_fac=1.5, 
                              proportion=0.7,
                              approx=True,
                              MLE=True):
    """
    Check that standalone functions reproduce same p-values
    as methods of `selectinf.randomized.lasso`
    """

    signal = np.sqrt(signal_fac * np.log(p)) / np.sqrt(n)
    X = np.random.standard_normal((n, p))
    T = np.random.exponential(1, size=(n,))
    S = np.random.choice([0,1], n, p=[0.2,0.8])

    cox_lasso = split_lasso.coxph(X, 
                                  T, 
                                  S,
                                  2 * np.ones(p) * np.sqrt(n),
                                  proportion)
    
    signs = cox_lasso.fit()
    nonzero = signs != 0

    cox_sel = rr.glm.cox(X[:,nonzero], T, S)

    cox_full = rr.glm.cox(X, T, S)

    refit_soln = cox_sel.solve(min_its=2000)
    padded_soln = np.zeros(p)
    padded_soln[nonzero] = refit_soln
    cox_full.solve(min_its=2000)
    
    full_hess = cox_full.hessian(padded_soln)
    selected_hess = full_hess[nonzero][:,nonzero]

    (observed_target, 
     cov_target, 
     cov_target_score, 
     alternatives) = selected_targets(cox_lasso.loglike, 
                                      None,
                                      nonzero,
                                      hessian=full_hess,
                                      dispersion=1)

    if nonzero.sum(): 
        if approx:
            approx_result = cox_lasso.approximate_grid_inference(observed_target, 
                                                                 cov_target, 
                                                                 cov_target_score)
            approx_pval = approx_result['pvalue']

            testval = approximate_normalizer_inference(proportion,
                                                       cox_lasso.initial_soln[nonzero],
                                                       refit_soln,
                                                       signs[nonzero],
                                                       selected_hess,
                                                       cox_lasso.feature_weights[nonzero])

            assert np.allclose(testval['pvalue'], approx_pval)

        else:
            approx_pval = np.empty(nonzero.sum())*np.nan

        if MLE:
            MLE_result = cox_lasso.selective_MLE(observed_target, 
                                                 cov_target, 
                                                 cov_target_score)[0]
            MLE_pval = MLE_result['pvalue']
        else:
            MLE_pval = np.empty(nonzero.sum())*np.nan

        # working under null here
        beta = np.zeros(p)

        testval = approximate_mle_inference(proportion,
                                            cox_lasso.initial_soln[nonzero],
                                            refit_soln,
                                            signs[nonzero],
                                            selected_hess,
                                            cox_lasso.feature_weights[nonzero])

        assert np.allclose(testval['pvalue'], MLE_pval)
        return approx_pval[beta[nonzero] == 0], MLE_pval[beta[nonzero] == 0], testval
    else:
        return [], []

def approximate_mle_inference(training_proportion,
                              training_betahat,
                              selected_beta_refit,
                              selected_signs,
                              selected_hessian,
                              selected_feature_weights,
                              level=0.9): 

    nselect = selected_hessian.shape[0]
    pi_s = training_proportion
    ratio = (1 - pi_s) / pi_s

    target_cov = np.linalg.inv(selected_hessian)
    cond_precision = selected_hessian / ratio
    cond_cov = target_cov * ratio
    selected_signs[np.isnan(selected_signs)] = 1 # for unpenalized
    cond_cov = cond_cov * selected_signs[None, :] * selected_signs[:, None]

    logdens_linear = target_cov * selected_signs[:,None] 
    cond_mean = selected_beta_refit * selected_signs - logdens_linear.dot(
                    selected_feature_weights *
                    selected_signs)
    linear_part = -np.identity(nselect)
    offset = np.zeros(nselect)

    target_score_cov = -np.identity(nselect)
    observed_target = selected_beta_refit
    
    result = selective_MLE(observed_target, 
                           target_cov,
                           target_score_cov, 
                           training_betahat * selected_signs,
                           cond_mean,
                           cond_cov,
                           logdens_linear,
                           linear_part,
                           offset,
                           level=level,
                           useC=True)[0]

    return result

def approximate_normalizer_inference(training_proportion,
                                     training_betahat,
                                     selected_beta_refit,
                                     selected_signs,
                                     selected_hessian,
                                     selected_feature_weights,
                                     alternatives=None,
                                     level=0.9): 

    nselect = selected_hessian.shape[0]
    pi_s = training_proportion
    ratio = (1 - pi_s) / pi_s

    target_cov = np.linalg.inv(selected_hessian)
    cond_precision = selected_hessian / ratio
    cond_cov = target_cov * ratio
    selected_signs[np.isnan(selected_signs)] = 1 # for unpenalized
    cond_cov = cond_cov * selected_signs[None, :] * selected_signs[:, None]

    logdens_linear = target_cov * selected_signs[:,None] 
    cond_mean = selected_beta_refit * selected_signs - logdens_linear.dot(
                    selected_feature_weights *
                    selected_signs)
    linear_part = -np.identity(nselect)
    offset = np.zeros(nselect)

    target_score_cov = -np.identity(nselect)
    observed_target = selected_beta_refit
    
    inverse_info = selective_MLE(observed_target, 
                                 target_cov,
                                 target_score_cov, 
                                 training_betahat * selected_signs,
                                 cond_mean,
                                 cond_cov,
                                 logdens_linear,
                                 linear_part,
                                 offset,
                                 level=level,
                                 useC=True)[1]

    G = approximate_grid_inference(observed_target,
                                   target_cov,
                                   target_score_cov,
                                   inverse_info,
                                   training_betahat * selected_signs,
                                   cond_mean,
                                   cond_cov,
                                   logdens_linear,
                                   linear_part,
                                   offset)

    return G.summary(alternatives=alternatives,
                     level=level)

