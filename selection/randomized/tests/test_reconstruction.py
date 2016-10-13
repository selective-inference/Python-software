from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.decorators import wait_for_return_value, register_report
import selection.tests.reports as reports

from selection.api import pairs_bootstrap_glm, multiple_queries, discrete_family, projected_langevin, glm_group_lasso_parametric
from selection.randomized.glm import split_glm_group_lasso
from selection.tests.instance import logistic_instance
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

from selection.randomized.multiple_queries import naive_confidence_intervals

@wait_for_return_value()
def test_reconstruction(s=3,
                        n=200,
                        p=50, 
                        snr=7,
                        rho=0.1,
                        split_frac=0.8,
                        lam_frac=0.7,
                        ndraw=10000, 
                        burnin=2000, 
                        bootstrap=True,
                        solve_args={'min_its':50, 'tol':1.e-10},
                        reference_known=False): 

    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)

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
    mv = multiple_queries([M_est])
    mv.solve()

    M_est.overall = M_est.overall
    nactive = np.sum(M_est.overall)

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(M_est.overall)[0]):

        active_set = np.nonzero(M_est.overall)[0]

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        boot_target, target_observed = pairs_bootstrap_glm(loss, M_est.overall)

        # testing the global null
        # constructing the intervals based on the samples of \bar{\beta}_E at the unpenalized MLE as a reference

        all_selected = np.arange(active_set.shape[0])
        target = lambda indices: boot_target(indices)[:nactive]
        target_observed = target_observed[:nactive]

        unpenalized_mle = restricted_Mest(loss, M_est.overall, solve_args=solve_args)

        alpha_mat = set_alpha_matrix(loss, M_est.overall)
        target_alpha = alpha_mat

        ## bootstrap
        reference_known = False
        if reference_known:
            reference = beta[M_est.overall] 
        else:
            reference = unpenalized_mle

        if bootstrap:
            target_sampler = mv.setup_bootstrapped_target(target,
                                                          target_observed,
                                                          n, target_alpha,
                                                          reference=reference) 

        else:
            target_sampler = mv.setup_target(target,
                                             target_observed, #reference=beta[M_est.overall])
                                             reference = unpenalized_mle)
            
        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin,
                                              keep_opt=True)
        
        stop

