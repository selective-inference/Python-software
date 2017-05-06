from __future__ import print_function
import numpy as np
import pandas as pd
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
                           glm_group_lasso_parametric)

from selection.randomized.query import (naive_confidence_intervals, naive_pvalues)

from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

@register_report(['truth', 'covered_clt', 'ci_length_clt',
                   'naive_pvalues','covered_naive', 'ci_length_naive'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_without_screening(s=30,
                        n=3000,
                        p=1000,
                        rho=0.,
                        snr=3.5,
                        lam_frac = 1.,
                        ndraw=10000,
                        burnin=2000,
                        loss='gaussian',
                        randomizer ='laplace',
                        randomizer_scale =1.,
                        scalings=False,
                        subgrad =True,
                        check_screen = False):

    if loss=="gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1, random_signs=False)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
        X_indep, y_indep, _, _, _ = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        loss_indep = rr.glm.gaussian(X_indep, y_indep)
    elif loss=="logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
        X_indep, y_indep, _, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr, random_signs=False)
        loss_indep = rr.glm.logistic(X_indep, y_indep)
    nonzero = np.where(beta)[0]

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,), scale=randomizer_scale)

    epsilon = 1. / np.sqrt(n)
    W = np.ones(p)*lam
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)
    M_est.solve()
    active_union = M_est._overall
    nactive = np.sum(active_union)
    print("nactive", nactive)
    active_set = np.nonzero(active_union)[0]
    print("active set", active_set)
    print("true nonzero", np.nonzero(beta)[0])

    views = [M_est]
    queries = multiple_queries(views)
    queries.solve()

    screened = False
    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        screened = True

    if check_screen==False or (check_screen==True and screened==True):

        #if nactive==s:
        #    return None

        if scalings: # try condition on some scalings
            M_est.condition_on_subgradient()
            M_est.condition_on_scalings()
        if subgrad:
            M_est.decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool), marginalizing_groups=np.ones(p, bool))


        boot_target1, boot_target_observed1 = pairs_bootstrap_glm(loss, active_union, inactive=~active_union)
        boot_target2, boot_target_observed2 = pairs_bootstrap_glm(loss_indep, active_union, inactive=~active_union)
        target_observed = (boot_target_observed1-boot_target_observed2)[:nactive]
        def _target(indices):
            return boot_target1(indices)[:nactive]-boot_target2(indices)[:nactive]

        form_covariances = glm_nonparametric_bootstrap(n, n)
        queries.setup_sampler(form_covariances)
        queries.setup_opt_state()

        target_sampler = queries.setup_target(_target,
                                              target_observed,
                                              reference=target_observed)

        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin)
        LU = target_sampler.confidence_intervals(target_observed,
                                                 sample=target_sample,
                                                 level=0.9)
        pivots = target_sampler.coefficient_pvalues(target_observed,
                                                    parameter=np.zeros(nactive),
                                                    sample=target_sample)

        #test_stat = lambda x: np.linalg.norm(x - beta[active_union])
        #observed_test_value = test_stat(target_observed)
        #pivots = target_sampler.hypothesis_test(test_stat,
        #                                       observed_test_value,
        #                                       alternative='twosided',
        #                                       parameter = beta[active_union],
        #                                       ndraw=ndraw,
        #                                       burnin=burnin,
        #                                       stepsize=None)

        true_vec = np.zeros(nactive)
        def coverage(LU):
            L, U = LU[:, 0], LU[:, 1]
            covered = np.zeros(nactive)
            ci_length = np.zeros(nactive)
            for j in range(nactive):
                if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                        covered[j] = 1
                ci_length[j] = U[j] - L[j]
            return covered, ci_length

        covered, ci_length = coverage(LU)
        LU_naive = naive_confidence_intervals(target_sampler, target_observed)
        covered_naive, ci_length_naive = coverage(LU_naive)
        naive_pvals = naive_pvalues(target_sampler, target_observed, true_vec)
        return pivots, covered, ci_length, naive_pvals, covered_naive, ci_length_naive


def report(niter=1, **kwargs):

    condition_report = reports.reports['test_without_screening']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    pkl_label = ''.join(["test_without_screening.pkl", "_", kwargs['loss'],"_",\
                         kwargs['randomizer'], ".pkl"])
    pdf_label = ''.join(["test_without_screening.pkl", "_", kwargs['loss'], "_", \
                         kwargs['randomizer'], ".pdf"])
    runs.to_pickle(pkl_label)
    runs_read = pd.read_pickle(pkl_label)

    fig = reports.pivot_plot_plus_naive(runs_read, color='b', label='no screening')


    fig.suptitle('Testing without screening', fontsize=20)
    fig.savefig(pdf_label)


if __name__ == '__main__':
    np.random.seed(500)
    kwargs = {'s':30, 'n':3000, 'p':1000, 'snr':3.5, 'rho':0, 'loss':'gaussian', 'randomizer':'gaussian',
                  'randomizer_scale':1.2, 'lam_frac':1.}
    report(niter=1, **kwargs)