import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests

import regreg.api as rr

from ...api import (randomization,
                    glm_group_lasso,
                    multiple_queries)
from ...tests.instance import (gaussian_instance,
                               logistic_instance)

from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.decorators import (wait_for_return_value, 
                                 set_seed_iftrue, 
                                 set_sampling_params_iftrue)

from ..query import naive_confidence_intervals, naive_pvalues
from ..M_estimator import restricted_Mest
from ..cv_view import CV_view
from ..glm import (glm_nonparametric_bootstrap,
                   pairs_bootstrap_glm)

if SMALL_SAMPLES:
    nboot = 10
else: 
    nboot = -1

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_cv(n=100, p=50, s=5, signal=7.5, K=5, rho=0.,
            randomizer = 'gaussian',
            randomizer_scale = 1.,
            scale1 = 0.1,
            scale2 = 0.2,
            lam_frac = 1.,
            glmnet = True,
            loss = 'gaussian',
            bootstrap = False,
            condition_on_CVR = True,
            marginalize_subgrad = True,
            ndraw = 10000,
            burnin = 2000,
            nboot = nboot):
    
    print(n,p,s, condition_on_CVR, scale1, scale2)
    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,),randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=randomizer_scale)

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1)
        glm_loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)
        glm_loss = rr.glm.logistic(X, y)

    epsilon = 1./np.sqrt(n)

    # view 1
    cv = CV_view(glm_loss, 
                 loss_label=loss, 
                 lasso_randomization=randomizer, 
                 epsilon=epsilon, 
                 scale1=scale1, 
                 scale2=scale2)
    if glmnet:
        try:
            cv.solve(glmnet=glmnet)
        except ImportError:
            cv.solve(glmnet=False)
    else:
        cv.solve(glmnet=False)

    # for the test make sure we also run the python code

    cv_py = CV_view(glm_loss, 
                    loss_label=loss, 
                    lasso_randomization=randomizer, 
                    epsilon=epsilon, 
                    scale1=scale1, 
                    scale2=scale2)
    cv_py.solve(glmnet=False)

    lam = cv.lam_CVR
    print("lam", lam)

    if condition_on_CVR:
        cv.condition_on_opt_state()
        lam = cv.one_SD_rule(direction="up")
        print("new lam", lam)

    # non-randomized Lasso, just looking how many vars it selects
    problem = rr.simple_problem(glm_loss, rr.l1norm(p, lagrange=lam))
    beta_hat = problem.solve()
    active_hat = beta_hat !=0
    print("non-randomized lasso ", active_hat.sum())

    # view 2
    W = lam_frac * np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    M_est = glm_group_lasso(glm_loss, epsilon, penalty, randomizer)

    if nboot > 0:
        cv.nboot = M_est.nboot = nboot

    mv = multiple_queries([cv, M_est])
    mv.solve()

    active_union = M_est._overall
    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    nonzero = np.where(beta)[0]

    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]
        true_vec = beta[active_union]

        if marginalize_subgrad == True:
            M_est.decompose_subgradient(conditioning_groups=np.zeros(p, bool),
                                         marginalizing_groups=np.ones(p, bool))

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
        sel_covered = np.zeros(nactive, np.bool)
        sel_length = np.zeros(nactive)

        LU_naive = naive_confidence_intervals(np.diag(target_cov), target_observed)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = np.zeros(nactive)
        naive_pvals = naive_pvalues(np.diag(target_cov), target_observed, true_vec)

        active_var = np.zeros(nactive, np.bool)

        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                sel_covered[j] = 1
            if (LU_naive[j, 0] <= true_vec[j]) and (LU_naive[j, 1] >= true_vec[j]):
                naive_covered[j] = 1
            sel_length[j] = U[j]-L[j]
            naive_length[j] = LU_naive[j,1]-LU_naive[j,0]
            active_var[j] = active_set[j] in nonzero

        q = 0.2
        BH_desicions = multipletests(pvalues, alpha=q, method="fdr_bh")[0]
        return sel_covered, sel_length, naive_pvals, naive_covered, naive_length, active_var, BH_desicions, active_var

