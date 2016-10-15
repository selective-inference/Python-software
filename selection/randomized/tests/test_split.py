from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES

from selection.api import multiple_queries, glm_target
from selection.randomized.glm import split_glm_group_lasso
from selection.tests.instance import logistic_instance

from selection.randomized.query import naive_confidence_intervals

@register_report(['mle', 'truth', 'pvalue', 'cover', 'naive_cover', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_split(s=3,
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

    M_est.selection_variable['variables'] = M_est.selection_variable['variables']
    nactive = np.sum(M_est.selection_variable['variables'])

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(M_est.selection_variable['variables'])[0]):

        active_set = np.nonzero(M_est.selection_variable['variables'])[0]

        if bootstrap:
            target_sampler, target_observed = glm_target(loss, 
                                                         M_est.selection_variable['variables'],
                                                         mv)

        else:
            target_sampler, target_observed = glm_target(loss, 
                                                         M_est.selection_variable['variables'],
                                                         mv,
                                                         bootstrap=True)

        reference_known = True
        if reference_known:
            reference = beta[M_est.selection_variable['variables']] 
        else:
            reference = target_observed

        target_sampler.reference = reference

        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin)


        LU = target_sampler.confidence_intervals(target_observed,
                                                 sample=target_sample).T

        LU_naive = naive_confidence_intervals(target_sampler, target_observed)

        pivots_mle = target_sampler.coefficient_pvalues(target_observed,
                                                        parameter=target_sampler.reference,
                                                        sample=target_sample)
        
        pivots_truth = target_sampler.coefficient_pvalues(target_observed,
                                                          parameter=beta[M_est.selection_variable['variables']],
                                                          sample=target_sample)
        
        true_vec = beta[M_est.selection_variable['variables']]

        pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                     parameter=np.zeros_like(true_vec),
                                                     sample=target_sample)

        L, U = LU

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

def report(niter=50, **kwargs):

    split_report = reports.reports['test_split']
    CLT_runs = reports.collect_multiple_runs(split_report['test'],
                                             split_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)
    kwargs['bootstrap'] = False
    fig = reports.pivot_plot(CLT_runs, color='b', label='CLT')

    kwargs['bootstrap'] = True
    bootstrap_runs = reports.collect_multiple_runs(split_report['test'],
                                                   split_report['columns'],
                                                   niter,
                                                   reports.summarize_all,
                                                   **kwargs)
    fig = reports.pivot_plot(bootstrap_runs, color='g', label='Bootstrap', fig=fig)
    fig.savefig('split_pivots.pdf') # will have both bootstrap and CLT on plot
