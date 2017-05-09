import numpy as np
import pandas as pd
import regreg.api as rr
from selection.api import (randomization,
                           glm_group_lasso,
                           multiple_queries,
                           glm_target)
from selection.tests.instance import (gaussian_instance,
                                      logistic_instance)

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
from selection.randomized.cv_view import CV_view
from statsmodels.sandbox.stats.multicomp import multipletests


@register_report(['truth', 'cover', 'ci_length_clt', 'naive_pvalues', 'naive_cover', 'ci_length_naive',
                    'active', 'BH_decisions', 'active_var'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_cv(n=100, p=50, s=0, snr=3.5, K=5, rho=0.,
             randomizer = 'gaussian',
             randomizer_scale = 1.,
             scale1 = 0.1,
             scale2 = 0.2,
             lam_frac = 1.,
             loss = 'gaussian',
             intervals = 'old',
             bootstrap = False,
             condition_on_CVR = True,
             marginalize_subgrad = True,
             ndraw = 10000,
             burnin = 2000):

    print(n,p,s, condition_on_CVR, scale1, scale2)
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
    cv = CV_view(glm_loss, loss_label=loss, lasso_randomization=randomizer, epsilon=epsilon,  scale1=scale1, scale2=scale2)
    cv.solve(glmnet=True)

    lam = cv.lam_CVR
    print("lam", lam)

    if condition_on_CVR:
        cv.condition_on_opt_state()
        lam = cv.one_SD_rule(direction="up")
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
    #mv = multiple_queries([M_est1])
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

        L, U = LU.T
        sel_covered = np.zeros(nactive, np.bool)
        sel_length = np.zeros(nactive)

        LU_naive = naive_confidence_intervals(target_sampler, target_observed)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = np.zeros(nactive)
        naive_pvals = naive_pvalues(target_sampler, target_observed, true_vec)

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
        return pivots_truth, sel_covered, sel_length, naive_pvals, naive_covered, naive_length, active_var, BH_desicions, active_var


def report(niter=50, **kwargs):
    np.random.seed(500)
    #kwargs = {'s': 0, 'n': 600, 'p': 100, 'snr': 3.5, 'bootstrap': False}
    intervals_report = reports.reports['test_cv']
    runs = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    pkl_label = ''.join([kwargs['loss'], "_", str(kwargs['condition_on_CVR']), "_", "test_cv.pkl"])
    pdf_label = ''.join([kwargs['loss'], "_", str(kwargs['condition_on_CVR']), "_", "test_cv.pdf"])
    runs.to_pickle(pkl_label)
    runs_read = pd.read_pickle(pkl_label)

    fig = reports.pivot_plot_plus_naive(runs_read)
    fig.suptitle("CV pivots", fontsize=20)
    fig.savefig(pdf_label)


if __name__ == '__main__':
    np.random.seed(500)
    kwargs = {'n': 600, 'p': 20, 's': 0, 'snr': 3.5, 'K': 5, 'rho': 0.,
              'randomizer': 'gaussian', 'randomizer_scale': 1.5,
              'scale1': 0.1, 'scale2': 0.1,  'lam_frac': 1.,
              'loss': 'logistic', 'intervals': 'old',
              'bootstrap': False, 'condition_on_CVR': True, 'marginalize_subgrad':  True}
    report(niter=1, **kwargs)