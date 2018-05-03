import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.instance import gaussian_instance
from ...tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue

from ..api import randomization 
from ..glm import (resid_bootstrap, 
                   glm_nonparametric_bootstrap,
                   fixedX_group_lasso)


@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_fixedX(ndraw=10000, burnin=2000): # nsim needed for decorator
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    X, Y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.1, signal=7)

    lam_frac = 1.
    lam = lam_frac * np.mean(np.fabs(X.T.dot(np.random.standard_normal((n, 50000)))).max(0)) * sigma
    W = np.ones(p) * lam
    epsilon = 1. / np.sqrt(n)

    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = fixedX_group_lasso(X, Y, epsilon, penalty, randomizer)
    M_est.solve()

    active_set = M_est.selection_variable['variables']
    nactive = active_set.sum()

    if set(nonzero).issubset(np.nonzero(active_set)[0]) and active_set.sum() > len(nonzero):

        selected_features = np.zeros(p, np.bool)
        selected_features[active_set] = True

        Xactive = X[:,active_set]
        unpenalized_mle = np.linalg.pinv(Xactive).dot(Y)

        form_covariances = glm_nonparametric_bootstrap(n, n)
        target_info, target_observed = resid_bootstrap(M_est.loss, active_set)

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
        active_set = np.nonzero(active_set)[0]

        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            active_var[j] = active_set[j] in nonzero

        return pvalues, covered, active_var

