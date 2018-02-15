"""
These tests exposes lower level functions than needed -- see test_convenience for simpler constructions
"""
from __future__ import print_function
import numpy as np, pandas as pd
from scipy.stats import norm as ndist

import regreg.api as rr

from ...tests.decorators import wait_for_return_value, register_report
import selection.tests.reports as reports
from ...tests.instance import logistic_instance

from ..api import randomization, multiple_queries, pairs_bootstrap_glm, glm_group_lasso, glm_nonparametric_bootstrap 
from ..glm import bootstrap_cov
from ...distributions.discrete_family import discrete_family
from ...sampling.langevin import projected_langevin
from ..reconstruct import reconstruct_internal

@register_report(['pvalue', 'active'])
@wait_for_return_value()
def test_overall_null_two_queries():
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, signal=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1. / np.sqrt(n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    # first randomization

    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    M_est1.solve()
    bootstrap_score1 = M_est1.setup_sampler(scaling=2.)

    # second randomization

    M_est2 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    M_est2.solve()
    bootstrap_score2 = M_est2.setup_sampler(scaling=2.)

    # we take target to be union of two active sets

    active = M_est1.selection_variable['variables'] + M_est2.selection_variable['variables']

    if set(nonzero).issubset(np.nonzero(active)[0]):
        boot_target, target_observed = pairs_bootstrap_glm(loss, active)

        # target are all true null coefficients selected

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        target_cov, cov1, cov2 = bootstrap_cov(sampler, boot_target, cross_terms=(bootstrap_score1, bootstrap_score2))

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        # is it enough only to bootstrap the inactive ones?
        # seems so...

        if not I:
            return None

        A1, b1 = M_est1.linear_decomposition(cov1[I], target_cov[I][:,I], target_observed[I])
        A2, b2 = M_est2.linear_decomposition(cov2[I], target_cov[I][:,I], target_observed[I])

        target_inv_cov = np.linalg.inv(target_cov[I][:,I])

        initial_state = np.hstack([target_observed[I],
                                   M_est1.observed_opt_state,
                                   M_est2.observed_opt_state])

        ntarget = len(I)
        target_slice = slice(0, ntarget)
        opt_slice1 = slice(ntarget, p + ntarget)
        opt_slice2 = slice(p + ntarget, 2*p + ntarget)

        def target_gradient(state):
            # with many samplers, we will add up the `target_slice` component
            # many target_grads
            # and only once do the Gaussian addition of full_grad

            target = state[target_slice]
            opt_state1 = state[opt_slice1]
            opt_state2 = state[opt_slice2]
            opt_linear1 = M_est1.opt_transform[0]
            arg1 = reconstruct_internal(target, (A1, b1))
            grad1 = M_est1.sampler.gradient(opt_state1)

            opt_linear2 = M_est2.opt_transform[0]
            arg2 = reconstruct_internal(target, (A2, b2))
            grad2 = M_est2.sampler.gradient(opt_state2)

            full_grad = np.zeros_like(state)
            full_grad[opt_slice1] = -opt_linear1.T.dot(grad1)
            full_grad[opt_slice2] = -opt_linear2.T.dot(grad2)
            full_grad[target_slice] -= A1.T.dot(grad1) + A2.T.dot(grad2)
            full_grad[target_slice] -= target_inv_cov.dot(target)

            return full_grad

        def target_projection(state):
            opt_state1 = state[opt_slice1]
            state[opt_slice1] = M_est1.sampler.projection(opt_state1)
            opt_state2 = state[opt_slice2]
            state[opt_slice2] = M_est2.sampler.projection(opt_state2)
            return state

        target_langevin = projected_langevin(initial_state,
                                             target_gradient,
                                             target_projection,
                                             .5 / (2*p + 1))


        Langevin_steps = 10000
        burning = 2000
        samples = []
        for i in range(Langevin_steps):
            target_langevin.next()
            if (i>=burning):
                samples.append(target_langevin.state[target_slice].copy())
                
        test_stat = lambda x: np.linalg.norm(x)
        observed = test_stat(target_observed[I])
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.ccdf(0, observed)
        return pval, False

@register_report(['pvalue', 'naive_pvalue', 'active'])
@wait_for_return_value()
def test_one_inactive_coordinate_handcoded():
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, signal=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W += lam * np.arange(p) / 200
    W[0] = 0
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print(lam)
    # our randomization

    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    M_est1.solve()
    bootstrap_score1 = M_est1.setup_sampler()

    active = M_est1.selection_variable['variables']
    if set(nonzero).issubset(np.nonzero(active)[0]):
        boot_target, target_observed = pairs_bootstrap_glm(loss, active)

        # target are all true null coefficients selected

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        target_cov, cov1 = bootstrap_cov(sampler, boot_target, cross_terms=(bootstrap_score1,))

        # have checked that covariance up to here agrees with other test_glm_langevin example

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        # is it enough only to bootstrap the inactive ones?
        # seems so...

        if not I:
            return None

        # take the first inactive one
        I = I[:1]
        A1, b1 = M_est1.linear_decomposition(cov1[I], target_cov[I][:,I], target_observed[I])

        print(I , 'I', target_observed[I])
        target_inv_cov = np.linalg.inv(target_cov[I][:,I])

        initial_state = np.hstack([target_observed[I],
                                   M_est1.observed_opt_state])

        ntarget = len(I)
        target_slice = slice(0, ntarget)
        opt_slice1 = slice(ntarget, p + ntarget)

        def target_gradient(state):
            # with many samplers, we will add up the `target_slice` component
            # many target_grads
            # and only once do the Gaussian addition of full_grad

            target = state[target_slice]
            opt_state1 = state[opt_slice1]

            opt_linear1 = M_est1.opt_transform[0]
            arg1 = reconstruct_internal(target, (A1, b1))
            grad1 = M_est1.sampler.gradient(opt_state1)

            full_grad = np.zeros_like(state)
            full_grad[opt_slice1] = -opt_linear1.T.dot(grad1)
            full_grad[target_slice] -= A1.T.dot(grad1)
            full_grad[target_slice] -= target_inv_cov.dot(target)

            return full_grad

        def target_projection(state):
            opt_state1 = state[opt_slice1]
            state[opt_slice1] = M_est1.sampler.projection(opt_state1)
            return state

        target_langevin = projected_langevin(initial_state,
                                             target_gradient,
                                             target_projection,
                                             1. / p)


        Langevin_steps = 10000
        burning = 2000
        samples = []
        for i in range(Langevin_steps + burning):
            target_langevin.next()
            if (i>burning):
                samples.append(target_langevin.state[target_slice].copy())
                
        test_stat = lambda x: x
        observed = test_stat(target_observed[I])
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.ccdf(0, observed)
        pval = 2 * min(pval, 1-pval)
        
        _i = I[0]
        naive_Z = target_observed[_i] / np.sqrt(target_cov[_i,_i])
        naive_pval = ndist.sf(np.fabs(naive_Z))
        naive_pval = 2 * min(naive_pval, 1 - naive_pval)
        print('naive Z', naive_Z, naive_pval)
        return pval, naive_pval, False


def report(niter=50):
    
    # these are all our null tests
    fn_names = ['test_overall_null_two_queries',
                'test_one_inactive_coordinate_handcoded']

    dfs = []
    for fn in fn_names:
        fn = reports.reports[fn]
        dfs.append(reports.collect_multiple_runs(fn['test'],
                                                 fn['columns'],
                                                 niter,
                                                 reports.summarize_all))
    dfs = pd.concat(dfs)

    fig = reports.pvalue_plot(dfs, colors=['r', 'g'])
    fig = reports.naive_pvalue_plot(dfs, fig=fig, colors=['k', 'b'])

    fig.savefig('Mest_pvalues.pdf') # will have both bootstrap and CLT on plot
