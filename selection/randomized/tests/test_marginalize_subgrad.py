from __future__ import print_function
import numpy as np

import regreg.api as rr
import selection.tests.reports as reports


from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.tests.decorators import (wait_for_return_value,
                                        set_seed_iftrue,
                                        set_sampling_params_iftrue,
                                        register_report)
import selection.tests.reports as reports

from selection.api import (randomization,
                           glm_group_lasso,
                           pairs_bootstrap_glm,
                           multiple_queries,
                           discrete_family,
                           projected_langevin,
                           glm_group_lasso_parametric,
                           glm_target)

from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)

from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

@register_report(['truth', 'covered_clt', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_marginalize(s=0,
                    n=600,
                    p=200,
                    rho=0.,
                    signal=3.5,
                    lam_frac = 2.5,
                    ndraw=10000,
                    burnin=2000,
                    loss='gaussian',
                    randomizer = 'gaussian',
                    randomizer_scale = 1.,
                    nviews=3,
                    scalings=False,
                    subgrad =True,
                    parametric=False,
                    intervals='old'):
    print(n,p,s)

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=randomizer_scale)

    if loss=="gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss=="logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    epsilon = 1. / np.sqrt(n)

    W = lam_frac*np.ones(p)*lam
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    views = []
    for i in range(nviews):
        if parametric==False:
            views.append(glm_group_lasso(loss, epsilon, penalty, randomizer))
        else:
            views.append(glm_group_lasso_parametric(loss, epsilon, penalty, randomizer))

    queries = multiple_queries(views)
    queries.solve()

    active_union = np.zeros(p, np.bool)
    for view in views:
        active_union += view.selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)

    nonzero = np.where(beta)[0]
    true_vec = beta[active_union]

    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        check_screen=True

        if nactive==s:
            return None

        if scalings: # try condition on some scalings
            for i in range(nviews):
                views[i].condition_on_subgradient()
                views[i].condition_on_scalings()
        if subgrad:
            for i in range(nviews):
               conditioning_groups = np.zeros(p,dtype=bool)
               conditioning_groups[:(p/2)] = True
               marginalizing_groups = np.zeros(p, dtype=bool)
               marginalizing_groups[(p/2):] = True
               views[i].decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool), marginalizing_groups=np.ones(p, bool))

        active_set = np.nonzero(active_union)[0]
        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False,
                                                     parametric=parametric)
                                                     #reference= beta[active_union])

        if intervals=='old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            LU = target_sampler.confidence_intervals(target_observed,
                                                     sample=target_sample,
                                                     level=0.9)
            pivots = target_sampler.coefficient_pvalues(target_observed,
                                                        parameter=true_vec,
                                                        sample=target_sample)
        elif intervals=='new':
            full_sample = target_sampler.sample(ndraw=ndraw,
                                                burnin=burnin,
                                                keep_opt=True)
            LU = target_sampler.confidence_intervals_translate(target_observed,
                                                           sample=full_sample,
                                                           level=0.9)
            pivots = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                    parameter=true_vec,
                                                                    sample=full_sample)

        #test_stat = lambda x: np.linalg.norm(x - beta[active_union])
        #observed_test_value = test_stat(target_observed)
        #pivots = target_sampler.hypothesis_test(test_stat,
        #                                       observed_test_value,
        #                                       alternative='twosided',
        #                                       parameter = beta[active_union],
        #                                       ndraw=ndraw,
        #                                       burnin=burnin,
        #                                       stepsize=None)

        def coverage(LU):
            L, U = LU[:, 0], LU[:, 1]
            covered = np.zeros(nactive)
            ci_length = np.zeros(nactive)

            for j in range(nactive):
                if check_screen:
                    if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                        covered[j] = 1
                else:
                    covered[j] = None
                ci_length[j] = U[j] - L[j]
            return covered, ci_length

        covered, ci_length = coverage(LU)
        LU_naive = naive_confidence_intervals(target_sampler, target_observed)
        covered_naive, ci_length_naive = coverage(LU_naive)
        naive_pvals = naive_pvalues(target_sampler, target_observed, true_vec)

        return pivots, covered, ci_length, naive_pvals, covered_naive, ci_length_naive

def report(niter=50, **kwargs):

    condition_report = reports.reports['test_marginalize']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pivot_plot_plus_naive(runs)
    #fig = reports.pivot_plot_2in1(runs,color='b', label='marginalized subgradient')
    fig.suptitle('Randomized Lasso marginalized subgradient')
    fig.savefig('marginalized_subgrad_pivots.pdf')


if __name__ == '__main__':
    report()
