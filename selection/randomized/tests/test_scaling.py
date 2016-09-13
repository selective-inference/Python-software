import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.randomized.api import randomization, multiple_views, pairs_bootstrap_glm, bootstrap_cov, glm_group_lasso
from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin

from selection.randomized.tests import wait_for_return_value, logistic_instance

#@wait_for_return_value
def test_logistic_many_targets(scaling=4., burnin=15000, ndraw=10000):
    DEBUG = False
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=scaling)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=14)
    X *= scaling

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = scaling**2

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est])
    mv.solve()

    active = M_est.overall
    nactive = active.sum()

    global_scaling = np.linalg.svd(X)[1].max()**2 # Lipschitz constant for gradient
    global_scaling = 1.
    sampler = lambda : np.random.choice(n, size=(n,), replace=True)

    def crude_target_scaling(_target_sampler):
        result = np.linalg.svd(_target_sampler.target_inv_cov)[1].max()
        for transform, objective, _boot in zip(_target_sampler.target_transform, _target_sampler.objectives, mv.score_bootstrap):
            result += np.linalg.svd(transform[0])[1].max()**2 * objective.randomization.lipschitz
            result += np.linalg.svd(objective.score_transform[0])[1].max()**2 * objective.randomization.lipschitz
            print np.linalg.svd(objective.score_transform[0])[1].max()**2 * objective.randomization.lipschitz, 'score'
            if DEBUG:
                print(transform[0][0,0], 'transform')
                print(_boot(sampler()), 'boot')
                print(objective.score_transform[0][0,0], 'score')
        return result * 2

    if set(nonzero).issubset(np.nonzero(active)[0]):

        if DEBUG:
            print M_est.initial_soln[:3] * scaling, scaling, 'first nonzero scaled'

        pvalues = []
        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        active_selected = A = [i for i in np.arange(active_set.shape[0]) if active_set[i] in nonzero]

        if not I:
            return None
        idx = I[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est.inactive, scaling=global_scaling)

        if DEBUG:
            print(boot_target(sampler())[-3:], 'boot target')

        mv.setup_sampler(sampler, scaling=global_scaling)

        # null saturated

        def null_target(indices):
            result = boot_target(indices)
            return result[idx]

        null_observed = np.zeros(1)
        null_observed[0] = target_observed[idx]

        target_sampler = mv.setup_target(null_target, null_observed)

        #target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2# should have something do with noise scale too

        print crude_target_scaling(target_sampler), 'crude'

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=burnin, ndraw=ndraw, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
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

        print crude_target_scaling(target_sampler), 'crude'

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=burnin, ndraw=ndraw, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
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
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=burnin, ndraw=ndraw, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
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
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=burnin, ndraw=ndraw, stepsize=1./crude_target_scaling(target_sampler)) # twosided by default
        pvalues.append(pval)

        return pvalues
