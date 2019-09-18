from __future__ import print_function, division
import numpy as np

import regreg.api as rr

from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.instance import (gaussian_instance, logistic_instance)
from ...tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue

from ..randomization import randomization

from ..M_estimator import restricted_Mest
from ..query import (naive_pvalues, naive_confidence_intervals)
from ..glm import (glm_group_lasso,
                   glm_nonparametric_bootstrap,
                   glm_parametric_covariance,
                   pairs_bootstrap_glm)

@set_seed_iftrue(SET_SEED, seed=20)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_intervals(s=0,
                   n=200,
                   p=10,
                   signal=7,
                   rho=0.,
                   lam_frac=6.,
                   ndraw=10000, 
                   burnin=2000, 
                   bootstrap=True,
                   loss='gaussian',
                   randomizer = 'laplace',
                   solve_args={'min_its':50, 'tol':1.e-10}):

    if randomizer =='laplace':
        randomizer = randomization.laplace((p,), scale=1.)
    elif randomizer=='gaussian':
        randomizer = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=1.)

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    nonzero = np.where(beta)[0]
    epsilon = 1./np.sqrt(n)

    W = lam_frac*np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    groups = np.concatenate([np.arange(10) for i in range(p//10)])

    penalty = rr.group_lasso(groups,
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)


    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)
    M_est.solve()


    active_union = M_est.selection_variable['variables']
    print("active set", np.nonzero(active_union)[0])
    nactive = np.sum(active_union)

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]
        true_vec = beta[active_union]

        selected_features = np.zeros(p, np.bool)
        selected_features[active_set] = True

        unpenalized_mle = restricted_Mest(M_est.loss, selected_features)

        form_covariances = glm_nonparametric_bootstrap(n, n)
        target_info, target_observed = pairs_bootstrap_glm(M_est.loss, selected_features, inactive=None)

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

        L, U = intervals.T

        LU_naive = naive_confidence_intervals(np.diag(target_cov), target_observed)

        ci_length_sel = np.zeros(nactive)
        covered = np.zeros(nactive, np.bool)
        naive_covered = np.zeros(nactive, np.bool)
        ci_length_naive = np.zeros(nactive)
        active_var = np.zeros(nactive, np.bool)
        
        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            ci_length_sel[j] = U[j]-L[j]
            if (LU_naive[j,0] <= true_vec[j]) and (LU_naive[j,1] >= true_vec[j]):
                naive_covered[j] = 1
            ci_length_naive[j]= LU_naive[j,1]-LU_naive[j,0]
            active_var[j] = active_set[j] in nonzero

        naive_pvals = naive_pvalues(np.diag(target_cov), target_observed, true_vec)

        return (pvalues, 
                covered, 
                ci_length_sel,
                naive_pvals, 
                naive_covered, 
                ci_length_naive, 
                active_var)

