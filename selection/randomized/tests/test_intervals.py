from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.instance import logistic_instance
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
import selection.tests.reports as reports

from selection.api import (randomization, 
                           glm_group_lasso, 
                           multiple_queries, 
                           glm_target)
from selection.randomized.M_estimator import restricted_Mest
from selection.randomized.query import naive_confidence_intervals

@register_report(['mle', 'truth', 'pvalue', 'cover', 'naive_cover', 'active'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_intervals(s=0,
                   n=200,
                   p=10,
                   snr=7,
                   rho=0.1,
                   lam_frac=1.2,
                   ndraw=10000, 
                   burnin=2000, 
                   bootstrap=True,
                   intervals='old',
                   randomizer = 'laplace',
                   solve_args={'min_its':50, 'tol':1.e-10}):
    if randomizer =='laplace':
        randomizer = randomization.laplace((p,), scale=1.)
    elif randomizer=='gaussian':
        randomizer = randomization.gaussian(np.identity(p))
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=1.)

    #randomizer =randomization.logistic((p,), scale=1.)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)

    nonzero = np.where(beta)[0]

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    #W[0] = 0 # use at least some unpenalized
    print(p)
    groups = np.concatenate((np.arange(3), np.arange(3), np.arange(4)))
    print("groups", groups)
    penalty = rr.group_lasso(groups,
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    #penalty = rr.group_lasso(np.arange(p),
    #                         weights=dict(zip(np.arange(p), W)), lagrange=1.)

    # first randomization
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    # second randomization
    # M_est2 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    # mv = multiple_queries([M_est1, M_est2])

    mv = multiple_queries([M_est1])
    mv.solve()

    active_union = M_est1.selection_variable['variables']
    print(active_union)
    nactive = np.sum(active_union)

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]
        true_vec = beta[active_union]

        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     mv,
                                                     bootstrap=bootstrap)

        if intervals == 'old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            LU = target_sampler.confidence_intervals(target_observed,
                                                     sample=target_sample,
                                                     level=0.9)
            pivots_mle = target_sampler.coefficient_pvalues(target_observed,
                                                            parameter=target_sampler.reference,
                                                            sample=target_sample)
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
            pivots_mle = target_sampler.coefficient_pvalues_translate(target_observed,
                                                            parameter=target_sampler.reference,
                                                            sample=full_sample)
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
        active_var = np.zeros(nactive, np.bool)
        
        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            if (LU_naive[j,0] <= true_vec[j]) and (LU_naive[j,1] >= true_vec[j]):
                naive_covered[j] = 1
            active_var[j] = active_set[j] in nonzero

        return pivots_mle, pivots_truth, pvalues, covered, naive_covered, active_var


def report(niter=10, **kwargs):

    kwargs= {'s': 0, 'n': 300, 'p': 10, 'snr': 7, 'bootstrap': False, 'randomizer': 'gaussian'}
    intervals_report = reports.reports['test_intervals']
    CLT_runs = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    #fig = reports.pivot_plot(CLT_runs, color='b', label='CLT')
    fig = reports.pivot_plot_2in1(CLT_runs, color='b', label='CLT')

    kwargs['bootstrap'] = True
    bootstrap_runs = reports.collect_multiple_runs(intervals_report['test'],
                                                   intervals_report['columns'],
                                                   niter,
                                                   reports.summarize_all,
                                                   **kwargs)

    #fig = reports.pivot_plot(bootstrap_runs, color='g', label='Bootstrap', fig=fig)
    fig = reports.pivot_plot_2in1(bootstrap_runs, color='g', label='Bootstrap', fig=fig)
    fig.savefig('intervals_pivots.pdf') # will have both bootstrap and CLT on plot

if __name__== '__main__':
    report()