import numpy as np

import regreg.api as rr
from ..api import (randomization,
                   glm_group_lasso,
                   multiple_queries)

from ...tests.instance import (gaussian_instance,
                                      logistic_instance)
from ...algorithms.sqrt_lasso import (sqlasso_objective,
                                      choose_lambda,
                                      l2norm_glm)

from ..query import naive_confidence_intervals, naive_pvalues
from ..M_estimator import restricted_Mest
from ..glm import (split_glm_group_lasso,
                   glm_nonparametric_bootstrap,
                   glm_parametric_covariance,
                   pairs_bootstrap_glm)

from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue

def choose_lambda_with_randomization(X, randomization, quantile=0.90, ndraw=10000):
    X = rr.astransform(X)
    n, p = X.output_shape[0], X.input_shape[0]
    E = np.random.standard_normal((n, ndraw))
    E /= np.sqrt(np.sum(E**2, 0))[None,:]
    dist1 = np.fabs(X.adjoint_map(E)).max(0)
    dist2 = np.fabs(randomization.sample((ndraw,))).max(0)
    return np.percentile(dist1+dist2, 100*quantile)

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_sqrt_lasso(n=500, p=20, s=3, signal=10, K=5, rho=0.,
                    randomizer = 'gaussian',
                    randomizer_scale = 1.,
                    scale1 = 0.1,
                    scale2 = 0.2,
                    lam_frac = 1.,
                    bootstrap = False,
                    condition_on_CVR = False,
                    marginalize_subgrad = True,
                    ndraw = 10000,
                    burnin = 2000):

    print(n,p,s)
    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,),randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=randomizer_scale)

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1)
    lam_nonrandom = choose_lambda(X)
    lam_random = choose_lambda_with_randomization(X, randomizer)
    loss = l2norm_glm(X, y)
    #sqloss = rr.glm.gaussian(X, y)
    epsilon = 1./n

    # non-randomized sqrt-Lasso, just looking how many vars it selects
    problem = rr.simple_problem(loss, rr.l1norm(p, lagrange=lam_nonrandom))
    beta_hat = problem.solve()
    active_hat = beta_hat !=0
    print("non-randomized sqrt-root Lasso active set", np.where(beta_hat)[0])
    print("non-randomized sqrt-lasso", active_hat.sum())

    # view 2
    W = lam_frac * np.ones(p) * lam_random
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1. / np.sqrt(n))
    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_queries([M_est])
    mv.solve()

    active_set = M_est._overall
    nactive = np.sum(active_set)

    if nactive==0:
        return None

    nonzero = np.where(beta)[0]
    if set(nonzero).issubset(np.nonzero(active_set)[0]):

        active_set = np.nonzero(active_set)[0]
        true_vec = beta[active_set]

        if marginalize_subgrad == True:
            M_est.decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool),
                                        marginalizing_groups=np.ones(p, bool))

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



