import functools

import numpy as np
from scipy.stats import norm as ndist
import matplotlib.pyplot as plt

import regreg.api as rr
from selection.api import (randomization,
                           glm_group_lasso,
                           multiple_queries,
                           glm_target)
import selection.api as sel
from selection.tests.instance import (gaussian_instance,
                                      logistic_instance)
from selection.randomized.glm import (pairs_bootstrap_glm,
                                      glm_nonparametric_bootstrap)
from selection.algorithms.lasso import (glm_sandwich_estimator,
                                        lasso)
from selection.constraints.affine import (constraints,
                                          stack)
from selection.randomized.query import naive_confidence_intervals

import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
from selection.randomized.cv_view import CV_view


@register_report(['truth', 'cover', 'naive_cover', 'ci_length_naive',
                    'active', 'BH_decisions', 'active_var'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_cv(n=500, p=20, s=0, snr=5, K=5, rho=0.,
             randomizer='gaussian',
             randomizer_scale = 0.8,
             lam_frac = 1.,
             loss = 'gaussian',
             intervals = 'old',
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

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        glm_loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        glm_loss = rr.glm.logistic(X, y)

    epsilon = 1./np.sqrt(n)
    # view 1
    cv = CV_view(glm_loss, lasso_randomization=randomizer, epsilon=epsilon, scale1=0.1, scale2=0.5)
    cv.solve()
    lam = cv.lam_CVR
    if condition_on_CVR:
        print("old lam", lam)
        cv.condition_on_opt_state()
        lam = cv.one_SD_rule()
        print("new lam", lam)

    # non-randomied Lasso, just looking how many vars it selects
    problem = rr.simple_problem(glm_loss, rr.l1norm(p, lagrange=lam))
    beta_hat = problem.solve()
    active_hat = beta_hat !=0
    print("non-randomized lasso ", active_hat.sum())

    # view 2
    W = lam_frac * np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    M_est1 = glm_group_lasso(glm_loss, epsilon, penalty, randomizer)

    mv = multiple_queries([cv, M_est1])
    mv.solve()

    #active = soln != 0
    active_union = M_est1._overall
    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    nonzero = np.where(beta)[0]
    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]
        true_vec = beta[active_union]

        if marginalize_subgrad == True:
            M_est1.decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool),
                                         marginalizing_groups=np.ones(p, bool))

        target_sampler, target_observed = glm_target(glm_loss,
                                                     active_union,
                                                     mv,
                                                     bootstrap=bootstrap)

        if intervals == 'old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            LU = target_sampler.confidence_intervals(target_observed,
                                                     sample=target_sample,
                                                     level=0.9)

            #pivots_mle = target_sampler.coefficient_pvalues(target_observed,
            #                                                parameter=target_sampler.reference,
            #                                                sample=target_sample)
            pivots_truth = target_sampler.coefficient_pvalues(target_observed,
                                                              parameter=true_vec,
                                                              sample=target_sample)
            pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                         parameter=np.zeros_like(true_vec),
                                                         sample=target_sample)
        else:
            full_sample = target_sampler.sample(ndraw=ndraw,
                                                burnin=burnin,
                                                keep_opt=True)
            LU = target_sampler.confidence_intervals_translate(target_observed,
                                                               sample=full_sample,
                                                               level=0.9)
            #pivots_mle = target_sampler.coefficient_pvalues_translate(target_observed,
            #                                                          parameter=target_sampler.reference,
            #                                                          sample=full_sample)
            pivots_truth = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                        parameter=true_vec,
                                                                        sample=full_sample)
            pvalues = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                   parameter=np.zeros_like(true_vec),
                                                                   sample=full_sample)

        LU_naive = naive_confidence_intervals(target_sampler, target_observed)

        L, U = LU.T

        covered = np.zeros(nactive, np.bool)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = np.zeros(nactive)
        active_var = np.zeros(nactive, np.bool)

        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            if (LU_naive[j, 0] <= true_vec[j]) and (LU_naive[j, 1] >= true_vec[j]):
                naive_covered[j] = 1
            naive_length[j] = LU_naive[j,1]-LU_naive[j,0]
            active_var[j] = active_set[j] in nonzero

        print("individual coverage", np.true_divide(covered.sum(),nactive))
        from statsmodels.sandbox.stats.multicomp import multipletests
        q = 0.1
        BH_desicions = multipletests(pvalues, alpha=q, method="fdr_bh")[0]

        return pivots_truth, covered, naive_covered, naive_length, active_var, BH_desicions, active_var


def report(niter=20, **kwargs):

    kwargs = {'s': 0, 'n': 3000, 'p': 1000, 'snr': 7, 'bootstrap': False}
    intervals_report = reports.reports['test_cv']
    CLT_runs = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    fig = reports.pivot_plot_2in1(CLT_runs, color='b', label='CV')
    fig.suptitle("CV pivots")
    fig.savefig('cv_pivots.pdf')


if __name__ == '__main__':
    report()