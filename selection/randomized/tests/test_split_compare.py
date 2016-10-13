from __future__ import print_function
import numpy as np

import regreg.api as rr

import selection.tests.reports as reports


from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.api import randomization, split_glm_group_lasso, pairs_bootstrap_glm, multiple_queries, discrete_family, projected_langevin, glm_group_lasso_parametric
from selection.tests.instance import logistic_instance
from selection.tests.decorators import wait_for_return_value, register_report
from selection.randomized.glm import standard_ci, standard_ci_sm, glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

from selection.randomized.multiple_queries import naive_confidence_intervals

@register_report(['pivots_clt', 'pivots_boot', 'covered_clt', 'ci_length_clt', 'covered_boot', 'ci_length_boot', 'covered_split', 'ci_length_split', 'active_var',
                  'covered_naive'])
@wait_for_return_value()
def test_split_compare(ndraw=10000, burnin=2000, nsim=None, solve_args={'min_its':50, 'tol':1.e-10}, check_screen =True): # nsim needed for decorator
    # s, n, p = 0, 200, 10
    s, n, p = 6, 300, 40

    randomizer = randomization.laplace((p,), scale=1.)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=5)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    m = int(0.8 * n)
    # first randomization
    M_est1 = split_glm_group_lasso(loss, epsilon, m, penalty)
    # second randomization
    # M_est2 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    # mv = multiple_queries([M_est1, M_est2])
    mv = multiple_queries([M_est1])
    mv.solve()

    active_union = M_est1.overall #+ M_est2.overall
    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    leftout_indices = M_est1.randomized_loss.saturated_loss.case_weights == 0

    screen = set(nonzero).issubset(np.nonzero(active_union)[0])

    if check_screen and not screen:
        return None

    if True:
        active_set = np.nonzero(active_union)[0]

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        boot_target, target_observed = pairs_bootstrap_glm(loss, active_union)
        #print("target",target_observed)
        #print(pairs_bootstrap_glm(loss, active_union))
        # testing the global null
        # constructing the intervals based on the samples of \bar{\beta}_E at the unpenalized MLE as a reference

        target_gn = lambda indices: boot_target(indices)[:nactive]
        target_observed_gn = target_observed[:nactive]

        unpenalized_mle = restricted_Mest(loss, M_est1.overall, solve_args=solve_args)

        alpha_mat = set_alpha_matrix(loss, active_union)
        target_alpha_gn = alpha_mat

        ## bootstrap
        target_sampler_gn_boot = mv.setup_bootstrapped_target(target_gn,
                                                              target_observed_gn,
                                                              n, target_alpha_gn,
                                                              reference = unpenalized_mle)

        target_sample_boot = target_sampler_gn_boot.sample(ndraw=ndraw,
                                                           burnin=burnin)

        LU_boot = target_sampler_gn_boot.confidence_intervals(unpenalized_mle,
                                                         sample=target_sample_boot)

        pivots_boot = target_sampler_gn_boot.coefficient_pvalues(unpenalized_mle,
                                                              parameter=beta[active_union],
                                                              sample=target_sample_boot)

        ## CLT plugin
        target_sampler_gn = mv.setup_target(target_gn,
                                            target_observed_gn,
                                            reference = unpenalized_mle)

        target_sample = target_sampler_gn.sample(ndraw=ndraw,
                                                 burnin=burnin)


        LU = target_sampler_gn.confidence_intervals(unpenalized_mle,
                                                    sample=target_sample)

        LU_naive = naive_confidence_intervals(target_sampler_gn, unpenalized_mle)

        if X.shape[0] - leftout_indices.sum() > nactive:
            LU_split = standard_ci(X, y, active_union, leftout_indices)
            LU_split_sm = standard_ci_sm(X, y, active_union, leftout_indices)
        else:
            LU_split = LU_split_sm = np.ones((nactive, 2)) * np.nan

        pivots = target_sampler_gn.coefficient_pvalues(unpenalized_mle,
                                                       parameter=beta[active_union],
                                                       sample=target_sample)

        true_vec = beta[active_union]

        def coverage(LU):
            L, U = LU[:,0], LU[:,1]
            covered = np.zeros(nactive)
            ci_length = np.zeros(nactive)

            for j in range(nactive):
                if check_screen:
                  if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                    covered[j] = 1
                else:
                    covered[j] = None
                ci_length[j] = U[j]-L[j]
            return covered, ci_length

        covered, ci_length = coverage(LU)
        covered_boot, ci_length_boot = coverage(LU_boot)
        covered_split, ci_length_split = coverage(LU_split)
        covered_naive, ci_length_naive = coverage(LU_naive)

        active_var = np.zeros(nactive, np.bool)
        for j in range(nactive):
            active_var[j] = active_set[j] in nonzero

        return pivots, pivots_boot, covered, ci_length, covered_boot, ci_length_boot, \
               covered_split, ci_length_split, active_var, covered_naive, ci_length_naive

def report(niter=50, **kwargs):

    split_report = reports.reports['test_split_compare']
    screened_results = reports.collect_multiple_runs(split_report['test'],
                                             split_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    fig = reports.boot_clt_plot(screened_results, color='b')
    fig.savefig('split_compare_pivots.pdf') # will have both bootstrap and CLT on plot

if __name__ == "__main__":
    report()

