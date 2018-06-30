from __future__ import print_function
import numpy as np

import regreg.api as rr

from ...tests.decorators import wait_for_return_value, set_sampling_params_iftrue
from ...tests.flags import SMALL_SAMPLES
from ...tests.instance import logistic_instance

from ..glm import (split_glm_group_lasso,
                   glm_nonparametric_bootstrap,
                   glm_parametric_covariance,
                   pairs_bootstrap_glm)
from ..M_estimator import restricted_Mest

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_split(s=3,
               n=200,
               p=50, 
               signal=7,
               rho=0.1,
               split_frac=0.8,
               lam_frac=0.7,
               ndraw=10000, 
               burnin=2000, 
               solve_args={'min_its':50, 'tol':1.e-10},
               reference_known=False): 

    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)

    m = int(split_frac * n)
    nonzero = np.where(beta)[0]

    loss = rr.glm.logistic(X, y)
    epsilon = 1. / np.sqrt(n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 2000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = split_glm_group_lasso(loss, epsilon, m, penalty)
    M_est.solve()

    M_est.selection_variable['variables'] 
    nactive = np.sum(M_est.selection_variable['variables'])

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(M_est.selection_variable['variables'])[0]):

        active_set = np.nonzero(M_est.selection_variable['variables'])[0]

        selected_features = np.zeros(p, np.bool)
        selected_features[active_set] = True

        unpenalized_mle = restricted_Mest(M_est.loss, selected_features)

        form_covariances = glm_nonparametric_bootstrap(n, n)
        boot_target, boot_target_observed = pairs_bootstrap_glm(M_est.loss, selected_features, inactive=None)
        target_info = boot_target

        cov_info = M_est.setup_sampler()
        target_cov, score_cov = form_covariances(target_info,  
                                                 cross_terms=[cov_info],
                                                 nsample=M_est.nboot)

        opt_sample = M_est.sampler.sample(ndraw,
                                          burnin)

        pvalues = M_est.sampler.coefficient_pvalues(unpenalized_mle, 
                                                    target_cov, 
                                                    score_cov, 
                                                    parameter=np.zeros(selected_features.sum()), 
                                                    sample=opt_sample)
        intervals = M_est.sampler.confidence_intervals(unpenalized_mle, target_cov, score_cov, sample=opt_sample)

        true_vec = beta[M_est.selection_variable['variables']] 

        L, U = intervals.T

        covered = np.zeros(nactive, np.bool)
        active_var = np.zeros(nactive, np.bool)

        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            active_var[j] = active_set[j] in nonzero

        return pvalues, covered, active_var

