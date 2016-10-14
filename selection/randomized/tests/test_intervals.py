from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.instance import logistic_instance
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
import selection.tests.reports as reports

from selection.api import randomization, glm_group_lasso, pairs_bootstrap_glm, multiple_queries, discrete_family, projected_langevin, glm_group_lasso_parametric
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

from selection.randomized.multiple_queries import naive_confidence_intervals

@register_report(['mle', 'truth', 'pvalue', 'cover', 'naive_cover', 'active'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_intervals(s=3,
                   n=200,
                   p=50, 
                   snr=7,
                   rho=0.1,
                   split_frac=0.8,
                   lam_frac=0.7,
                   ndraw=10000, 
                   burnin=2000, 
                   bootstrap=True,
                   solve_args={'min_its':50, 'tol':1.e-10}):

    randomizer = randomization.laplace((p,), scale=1.)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)

    nonzero = np.where(beta)[0]

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    # first randomization
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    # second randomization
    # M_est2 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    # mv = multiple_queries([M_est1, M_est2])
    mv = multiple_queries([M_est1])
    mv.solve()

    active_union = M_est1.overall #+ M_est2.overall
    nactive = np.sum(active_union)

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        boot_target, target_observed = pairs_bootstrap_glm(loss, active_union)

        # testing the global null
        # constructing the intervals based on the samples of \bar{\beta}_E at the unpenalized MLE as a reference
        all_selected = np.arange(active_set.shape[0])
        target_gn = lambda indices: boot_target(indices)[:nactive]
        target_observed_gn = target_observed[:nactive]

        unpenalized_mle = restricted_Mest(loss, M_est1.overall, solve_args=solve_args)

        ## bootstrap

        if bootstrap:
            alpha_mat = set_alpha_matrix(loss, active_union)
            target_alpha_gn = alpha_mat
            target_sampler_gn = mv.setup_bootstrapped_target(target_gn,
                                                             target_observed_gn,
                                                             n, target_alpha_gn,
                                                             reference = unpenalized_mle)
        else:
            target_sampler_gn = mv.setup_target(target_gn,
                                                target_observed_gn,
                                                reference = unpenalized_mle)

        target_sample = target_sampler_gn.sample(ndraw=ndraw,
                                                 burnin=burnin)


        LU = target_sampler_gn.confidence_intervals(unpenalized_mle,
                                                    sample=target_sample)

        LU_naive = naive_confidence_intervals(target_sampler_gn, unpenalized_mle)

        pivots_mle = target_sampler_gn.coefficient_pvalues(unpenalized_mle,
                                                           parameter=target_sampler_gn.reference,
                                                           sample=target_sample)

        pivots_truth = target_sampler_gn.coefficient_pvalues(unpenalized_mle, 
                                                             parameter=beta[active_union],
                                                             sample=target_sample)
        true_vec = beta[active_union]
        pvalues = target_sampler_gn.coefficient_pvalues(unpenalized_mle,
                                                        parameter=np.zeros_like(true_vec),
                                                        sample=target_sample)

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

def report(niter=50, **kwargs):

    intervals_report = reports.reports['test_intervals']
    CLT_runs = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)
    kwargs['bootstrap'] = True
    fig = reports.pivot_plot(CLT_runs, color='b', label='Bootstrap')

    kwargs['bootstrap'] = False
    bootstrap_runs = reports.collect_multiple_runs(intervals_report['test'],
                                                   intervals_report['columns'],
                                                   niter,
                                                   reports.summarize_all,
                                                   **kwargs)

    fig = reports.pivot_plot(bootstrap_runs, color='g', label='CLT', fig=fig)
    fig.savefig('intervals_pivots.pdf') # will have both bootstrap and CLT on plot
