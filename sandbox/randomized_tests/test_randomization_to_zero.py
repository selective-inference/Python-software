from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
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
                           glm_group_lasso_parametric)

from selection.randomized.glm import (glm_parametric_covariance, 
                                      glm_nonparametric_bootstrap, 
                                      restricted_Mest, 
                                      set_alpha_matrix,
                                      target as glm_target)

import matplotlib.pyplot as plt
import statsmodels.api as sm

@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=100, burnin=100)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value(max_tries=200)
def test_multiple_queries_small(ndraw=10000, burnin=2000, nsim=None): # nsim needed for decorator
    s, n, p = 2, 100, 10

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, signal=3)

    nonzero = np.where(beta)[0]
    lam_frac = .6

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    # first randomization
    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_queries([M_est])
    mv.solve()

    active_union = M_est.selection_variable['variables'] 
    nactive = np.sum(active_union)
    print("nactive", nactive)

    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        if nactive==s:
            return None

        active_set = np.nonzero(active_union)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        if not I:
            return None

        inactive_indicators_mat = np.zeros((len(inactive_selected),nactive))
        j = 0
        for i in range(nactive):
            if active_set[i] not in nonzero:
                inactive_indicators_mat[j,i] = 1
                j+=1

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        boot_target, target_observed = pairs_bootstrap_glm(loss, active_union)
        inactive_target = lambda indices: boot_target(indices)[inactive_selected]
        inactive_observed = target_observed[inactive_selected]
        # param_cov = _parametric_cov_glm(loss, active_union)

        alpha_mat = set_alpha_matrix(loss, active_union)
        # target = target_alpha\times alpha+reference_vec
        target_alpha = np.dot(inactive_indicators_mat, alpha_mat) 

        target_sampler = mv.setup_bootstrapped_target(inactive_target, inactive_observed, target_alpha)

        test_stat = lambda x: np.linalg.norm(x)
        pval = target_sampler.hypothesis_test(test_stat, 
                                              np.linalg.norm(inactive_observed), 
                                              alternative='twosided',
                                              ndraw=ndraw,
                                              burnin=burnin)


        # testing the global null
        all_selected = np.arange(active_set.shape[0])
        target_gn = lambda indices: boot_target(indices)[:nactive]
        target_observed_gn = target_observed[:nactive]

        target_alpha_gn = alpha_mat

        target_sampler_gn = mv.setup_bootstrapped_target(target_gn, target_observed_gn, target_alpha_gn, reference = beta[active_union])
        test_stat_boot_gn = lambda x: np.linalg.norm(x)
        observed_test_value = np.linalg.norm(target_observed_gn-beta[active_union])
        pval_gn = target_sampler_gn.hypothesis_test(test_stat_boot_gn, 
                                                    observed_test_value, 
                                                    alternative='twosided',
                                                    ndraw=ndraw,
                                                    burnin=burnin)

        return [pval, pval_gn], [False, False]


@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=100, burnin=100)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value(max_tries=300)
def test_multiple_queries_individual_coeff_small(ndraw=10000, 
                                                 burnin=2000, 
                                                 bootstrap=True):
    s, n, p = 3, 100, 20

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, true_active = logistic_instance(n=n, p=p, s=s, rho=0, signal=20.)

    nonzero = np.where(beta)[0]
    lam_frac = 1.2

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    # randomization
    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)
    mv = multiple_queries([M_est])
    mv.solve()

    active_vars = M_est.selection_variable['variables'] 

    nactive = np.sum(active_vars)
    active_set = np.nonzero(active_vars)[0]

    pvalues = []
    true_beta = beta[active_vars]

    if set(nonzero).issubset(active_set):

        for j in range(nactive):

            print(j)
            subset = np.zeros(p, np.bool)
            subset[active_set[j]] = True
            target_sampler, target_observed = glm_target(loss,
                                                         active_vars,
                                                         mv,
                                                         subset=subset,
                                                         bootstrap=bootstrap,
                                                         reference=np.zeros((1,)))

            test_stat = lambda x: x 

            pval = target_sampler.hypothesis_test(test_stat,
                                                  target_observed,
                                                  alternative='twosided',
                                                  ndraw=ndraw,
                                                  burnin=burnin)
            pvalues.append(pval)
        return pvalues, [active_set[j] in nonzero for j in range(nactive)]

@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=100, burnin=100)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_parametric_covariance_small(ndraw=10000, burnin=2000, nsim=None): # nsim needed for decorator
    s, n, p = 3, 100, 10

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, signal=15)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    # first randomization
    M_est1 = glm_group_lasso_parametric(loss, epsilon, penalty, randomizer)
    # second randomization
    M_est2 = glm_group_lasso_parametric(loss, epsilon, penalty, randomizer)

    mv = multiple_queries([M_est1, M_est2])
    mv.solve()

    target = M_est1.selection_variable['variables'].copy()
    if target[-1] or M_est2.selection_variable['variables'][-1]:
        return None
    if target[-2] or M_est2.selection_variable['variables'][-2]:
        return None
    # we should check they are different sizes
    target[-2:] = 1

    if set(nonzero).issubset(np.nonzero(target)[0]):

        form_covariances = glm_parametric_covariance(loss)
        mv.setup_sampler(form_covariances)

        target_observed = restricted_Mest(loss, target)
        linear_func = np.zeros((2,target_observed.shape[0]))
        linear_func[0,-1] = 1. # we know this one is null
        linear_func[1,-2] = 1. # also null

        target_observed = linear_func.dot(target_observed)
        target_sampler = mv.setup_target((target, linear_func), target_observed, parametric=True)

        test_stat = lambda x: np.linalg.norm(x)
        pval = target_sampler.hypothesis_test(test_stat, 
                                              test_stat(target_observed), 
                                              alternative='greater',
                                              ndraw=ndraw,
                                              burnin=burnin)

        return pval, False

def report(niter=50, **kwargs):
    # these are all our null tests
    fn_names = ['test_parametric_covariance_small',
                'test_multiple_queries_small',
                'test_multiple_queries_individual_coeff_small']

    dfs = []
    for fn in fn_names:
        fn = reports.reports[fn]
        dfs.append(reports.collect_multiple_runs(fn['test'],
                                                 fn['columns'],
                                                 niter,
                                                 reports.summarize_all))
    dfs = pd.concat(dfs)

    fig = reports.pvalue_plot(dfs, colors=['r', 'g'])

    fig.savefig('randomization_to_zero_pvalues.pdf') # will have both bootstrap and CLT on plot
