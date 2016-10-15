from __future__ import print_function
import numpy as np
import pandas as pd
import regreg.api as rr
import selection.tests.reports as reports


from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.instance import logistic_instance
from selection.tests.decorators import (wait_for_return_value, 
                                        set_seed_iftrue, 
                                        set_sampling_params_iftrue,
                                        register_report)
import selection.tests.reports as reports

from selection.api import randomization, glm_group_lasso, pairs_bootstrap_glm, multiple_queries, discrete_family, projected_langevin, glm_group_lasso_parametric, glm_target
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_multiple_queries(ndraw=10000, burnin=2000, bootstrap=True, test = 'selected zeros'): 
    s, n, p = 0, 600, 10

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=3)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    view = []
    nview = 5
    for i in range(nview):
        view.append(glm_group_lasso(loss, epsilon, penalty, randomizer))


    mv = multiple_queries(view)
    mv.solve()

    active_union = np.zeros(p, np.bool)
    for i in range(nview):
        active_union += view[i].selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)

    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        if nactive==s:
            return None

        active_set = np.nonzero(active_union)[0]

        if test == 'selected zeros':
            inactive_selected = [active_union[i] and i not in nonzero for i in range(p)]
            target_sampler, target_observed = glm_target(loss,
                                                         active_union,
                                                         mv,
                                                         subset=inactive_selected,
                                                         bootstrap=bootstrap)
        else:
            target_sampler, target_observed = glm_target(loss,
                                                         active_union,
                                                         mv,
                                                         bootstrap=bootstrap)

        test_stat = lambda x: np.linalg.norm(x)
        observed_test_value = test_stat(target_observed)
        pivot = target_sampler.hypothesis_test(test_stat,
                                               observed_test_value,
                                               alternative='twosided',
                                               ndraw=ndraw,
                                               burnin=burnin)

        return [pivot], [False]

@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=100, burnin=100)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_multiple_queries_individual_coeff(ndraw=10000, burnin=2000):
    s, n, p = 3, 120, 10

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=3)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    view = []
    nview = 5
    for i in range(nview):
        view.append(glm_group_lasso(loss, epsilon, penalty, randomizer))

    mv = multiple_queries(view)
    mv.solve()

    active_union = np.zeros(p, np.bool)
    for i in range(nview):
        active_union += view[i].selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)
    active_set = np.nonzero(active_union)[0]

    pvalues = []
    true_beta = beta[active_union]
    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        for j in range(nactive):

            subset = np.zeros(p, np.bool)
            subset[j] = True
            target_sampler, target_observed = glm_target(loss,
                                                         active_union,
                                                         mv,
                                                         subset=subset)
            test_stat = lambda x: np.atleast_1d(x)

            pval = target_sampler.hypothesis_test(test_stat,
                                                  np.atleast_1d(target_observed-true_beta[j]),
                                                  alternative='twosided',
                                                  ndraw=ndraw,
                                                  burnin=burnin)
            pvalues.append(pval)

        return pvalues, [active_set[j] in nonzero for j in range(nactive)]


@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=100, burnin=100)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value(max_tries=200)
def test_parametric_covariance(ndraw=10000, burnin=2000): 
    s, n, p = 3, 120, 10

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=12)

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
        target_sampler = mv.setup_target((target, linear_func), target_observed)

        test_stat = lambda x: np.linalg.norm(x)
        pval = target_sampler.hypothesis_test(test_stat,
                                              test_stat(target_observed),
                                              alternative='greater',
                                              ndraw=ndraw,
                                              burnin=burnin)

        return [pval], [False]

def report(niter=50):
    
    # these are all our null tests
    fn_names = ['test_parametric_covariance',
                'test_multiple_queries',
                'test_multiple_queries_individual_coeff']

    dfs = []
    for fn in fn_names:
        fn = reports.reports[fn]
        dfs.append(reports.collect_multiple_runs(fn['test'],
                                                 fn['columns'],
                                                 niter,
                                                 reports.summarize_all))
    dfs = pd.concat(dfs)

    fig = reports.pvalue_plot(dfs, colors=['r', 'g'])

    fig.savefig('multiple_queries_pvalues.pdf') # will have both bootstrap and CLT on plot

