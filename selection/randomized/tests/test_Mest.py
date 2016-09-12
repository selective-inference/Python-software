import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.randomized.api import randomization, multiple_views, pairs_bootstrap_glm, bootstrap_cov, glm_group_lasso
from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin

from selection.randomized.tests import wait_for_return_value, logistic_instance

@wait_for_return_value
def test_overall_null_two_views():
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)

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

    active = M_est1.overall + M_est2.overall

    if set(nonzero).issubset(np.nonzero(active)[0]):
        boot_target, target_observed = pairs_bootstrap_glm(loss, active)

        # target are all true null coefficients selected

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        target_cov, cov1, cov2 = bootstrap_cov(sampler, boot_target, cross_terms=(bootstrap_score1, bootstrap_score2))

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        # is it enough only to bootstrap the inactive ones?
        # seems so...

        A1, b1 = M_est1.condition(cov1[I], target_cov[I][:,I], target_observed[I])
        A2, b2 = M_est2.condition(cov2[I], target_cov[I][:,I], target_observed[I])

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
            target_grad1 = M_est1.randomization_gradient(target, (A1, b1), opt_state1)
            target_grad2 = M_est2.randomization_gradient(target, (A2, b2), opt_state2)

            full_grad = np.zeros_like(state)
            full_grad[opt_slice1] = -target_grad1[1]
            full_grad[opt_slice2] = -target_grad2[1]
            full_grad[target_slice] -= target_grad1[0] + target_grad2[0]
            full_grad[target_slice] -= target_inv_cov.dot(target)

            return full_grad

        def target_projection(state):
            opt_state1 = state[opt_slice1]
            state[opt_slice1] = M_est1.projection(opt_state1)
            opt_state2 = state[opt_slice2]
            state[opt_slice2] = M_est2.projection(opt_state2)
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
        return pval

def test_one_inactive_coordinate_handcoded(seed=None):
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W += lam * np.arange(p) / 200
    W[0] = 0
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    M_est1.solve()
    bootstrap_score1 = M_est1.setup_sampler()

    active = M_est1.overall
    if set(nonzero).issubset(np.nonzero(active)[0]):
        boot_target, target_observed = pairs_bootstrap_glm(loss, active)

        # target are all true null coefficients selected

        if seed is not None:
            np.random.seed(seed)
        sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        target_cov, cov1 = bootstrap_cov(sampler, boot_target, cross_terms=(bootstrap_score1,))

        # have checked that covariance up to here agrees with other test_glm_langevin example

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        # is it enough only to bootstrap the inactive ones?
        # seems so...

        if not I:
            return None, None

        # take the first inactive one
        I = I[:1]
        A1, b1 = M_est1.condition(cov1[I], target_cov[I][:,I], target_observed[I])

        print I, 'I', target_observed[I]
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
            target_grad1 = M_est1.randomization_gradient(target, (A1, b1), opt_state1)

            full_grad = np.zeros_like(state)
            full_grad[opt_slice1] = -target_grad1[1]
            full_grad[target_slice] -= target_grad1[0] 
            full_grad[target_slice] -= target_inv_cov.dot(target)

            return full_grad

        def target_projection(state):
            opt_state1 = state[opt_slice1]
            state[opt_slice1] = M_est1.projection(opt_state1)
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
        return pval, naive_pval
    else:
        return None, None

@wait_for_return_value
def test_logistic_selected_inactive_coordinate(seed=None):
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est1])
    mv.solve()

    active = M_est1.overall
    nactive = active.sum()
    scaling = np.linalg.svd(X)[1].max()**2

    if set(nonzero).issubset(np.nonzero(active)[0]):

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        if not I:
            return None

        idx = I[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est1.inactive)


        def null_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        null_observed = np.zeros(M_est1.inactive.sum() + 1)
        null_observed[0] = target_observed[idx]
        null_observed[1:] = target_observed[nactive:]

        # the null_observed[1:] is only used as a
        # starting point for chain -- could be 0

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        mv.setup_sampler(sampler, scaling=scaling)
        target_sampler = mv.setup_target(null_target, null_observed, target_set=[0])
        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=10000, ndraw=10000, stepsize=1./scaling) # twosided by default
        return pval

@wait_for_return_value
def test_logistic_saturated_inactive_coordinate(seed=None):
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est1])
    mv.solve()

    active = M_est1.overall
    nactive = active.sum()
    if set(nonzero).issubset(np.nonzero(active)[0]):

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        if not I:
            return None
        idx = I[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est1.inactive)

        def null_target(indices):
            result = boot_target(indices)
            return result[idx]

        null_observed = np.zeros(1)
        null_observed[0] = target_observed[idx]

        # the null_observed[1:] is only used as a
        # starting point for chain -- could be 0
        # null_observed[1:] = target_observed[nactive:]

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        mv.setup_sampler(sampler)
        target_sampler = mv.setup_target(null_target, null_observed)

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=10000, ndraw=10000) # twosided by default
        return pval

@wait_for_return_value
def test_logistic_selected_active_coordinate(seed=None):
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est1])
    mv.solve()

    active = M_est1.overall
    nactive = active.sum()
    if set(nonzero).issubset(np.nonzero(active)[0]):

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        active_selected = A = [i for i in np.arange(active_set.shape[0]) if active_set[i] in nonzero]

        idx = A[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est1.inactive)

        def active_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        active_observed = np.zeros(M_est1.inactive.sum() + 1)
        active_observed[0] = target_observed[idx]
        active_observed[1:] = target_observed[nactive:]

        # the active_observed[1:] is only used as a
        # starting point for chain -- could be 0
        # active_observed[1:] = target_observed[nactive:]

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        mv.setup_sampler(sampler)
        target_sampler = mv.setup_target(active_target, active_observed, target_set=[0])
        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=10000, ndraw=10000) # twosided by default
        return pval

@wait_for_return_value
def test_logistic_saturated_active_coordinate(seed=None):
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est1])
    mv.solve()

    active = M_est1.overall
    nactive = active.sum()
    if set(nonzero).issubset(np.nonzero(active)[0]):

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        active_selected = A = [i for i in np.arange(active_set.shape[0]) if active_set[i] in nonzero]

        idx = A[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est1.inactive)

        def active_target(indices):
            result = boot_target(indices)
            return result[idx]

        active_observed = np.zeros(1)
        active_observed[0] = target_observed[idx]

        # the active_observed[1:] is only used as a
        # starting point for chain -- could be 0
        # active_observed[1:] = target_observed[nactive:]

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        mv.setup_sampler(sampler)
        target_sampler = mv.setup_target(active_target, active_observed)
        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=10000, ndraw=10000) # twosided by default
        return pval

@wait_for_return_value
def test_logistic_many_targets():
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est])
    mv.solve()

    active = M_est.overall
    nactive = active.sum()

    global_scaling = np.linalg.svd(X)[1].max()**2

    def crude_target_scaling(_target_sampler):
        result = np.linalg.svd(_target_sampler.target_inv_cov)[1].max()
        for transform, objective in zip(_target_sampler.target_transform, _target_sampler.objectives):
            result += np.linalg.svd(transform[0])[1].max()**2 * objective.randomization.lipschitz
            result += np.linalg.svd(objective.score_transform[0])[1].max()**2 * objective.randomization.lipschitz
        return result

    if set(nonzero).issubset(np.nonzero(active)[0]):

        pvalues = []
        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        active_selected = A = [i for i in np.arange(active_set.shape[0]) if active_set[i] in nonzero]

        if not I:
            return None
        idx = I[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est.inactive)

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        mv.setup_sampler(sampler, scaling=global_scaling)

        # null saturated

        def null_target(indices):
            result = boot_target(indices)
            return result[idx]

        null_observed = np.zeros(1)
        null_observed[0] = target_observed[idx]

        target_sampler = mv.setup_target(null_target, null_observed)
        #target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2# should have something do with noise scale too

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=10000, ndraw=8000, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
        pvalues.append(pval)

        # null selected

        def null_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        null_observed = np.zeros_like(null_target(range(n)))
        null_observed[0] = target_observed[idx]
        null_observed[1:] = target_observed[nactive:]

        target_sampler = mv.setup_target(null_target, null_observed, target_set=[0])
        target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2# should have something do with noise scale too

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=10000, ndraw=8000, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
        pvalues.append(pval)

        # true saturated

        idx = A[0]

        def active_target(indices):
            result = boot_target(indices)
            return result[idx]

        active_observed = np.zeros(1)
        active_observed[0] = target_observed[idx]

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        target_sampler = mv.setup_target(active_target, active_observed)
        target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2# should have something do with noise scale too

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=10000, ndraw=8000, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
        pvalues.append(pval)

        # true selected

        def active_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        active_observed = np.zeros_like(active_target(range(n)))
        active_observed[0] = target_observed[idx]
        active_observed[1:] = target_observed[nactive:]

        target_sampler = mv.setup_target(active_target, active_observed, target_set=[0])
        target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2 # should have something do with noise scale too

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=10000, ndraw=8000, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
        pvalues.append(pval)

        return pvalues
