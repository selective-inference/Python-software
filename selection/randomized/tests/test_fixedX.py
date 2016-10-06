import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests import SMALL_SAMPLES, SET_SEED
from selection.randomized.randomization import randomization
from selection.randomized.multiple_views import multiple_views
from selection.randomized.glm import resid_bootstrap, fixedX_group_lasso, glm_nonparametric_bootstrap 
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=100, burnin=100)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_gaussian_many_targets(ndraw=10000, burnin=2000, nsim=None): # nsim needed for decorator
    s, n, p = 5, 200, 20 

    randomizer = randomization.laplace((p,), scale=1.)
    X, Y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.1, snr=7)

    lam_frac = 1.
    lam = lam_frac * np.mean(np.fabs(X.T.dot(np.random.standard_normal((n, 50000)))).max(0)) * sigma
    W = np.ones(p) * lam
    epsilon = 1. / np.sqrt(n)

    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = fixedX_group_lasso(X, Y, epsilon, penalty, randomizer)

    mv = multiple_views([M_est])
    mv.solve()

    active = M_est.overall
    nactive = active.sum()

    if set(nonzero).issubset(np.nonzero(active)[0]) and active.sum() > len(nonzero):

        pvalues = []
        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        active_selected = A = [i for i in np.arange(active_set.shape[0]) if active_set[i] in nonzero]

        idx = I[0]
        boot_target, target_observed = resid_bootstrap(M_est.loss, active)

        X_active = X[:,active]
        beta_hat = np.linalg.pinv(X_active).dot(Y)
        resid_hat = Y - X_active.dot(beta_hat)
        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        # null saturated

        def null_target(Y_star):
            result = boot_target(Y_star)
            return result[idx]

        null_observed = np.zeros(1)
        null_observed[0] = target_observed[idx]

        target_sampler = mv.setup_target(null_target, null_observed)

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=burnin, ndraw=ndraw) # twosided by default
        pvalues.append(pval)

        # null selected

        def null_target(Y_star):
            result = boot_target(Y_star)
            return np.hstack([result[idx], result[nactive:]])

        null_observed = np.zeros_like(null_target(np.random.standard_normal(n)))
        null_observed[0] = target_observed[idx]
        null_observed[1:] = target_observed[nactive:]

        target_sampler = mv.setup_target(null_target, null_observed, target_set=[0])

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, null_observed, burnin=burnin, ndraw=ndraw) # twosided by default
        pvalues.append(pval)

        # true saturated

        idx = A[0]

        def active_target(Y_star):
            result = boot_target(Y_star)
            return result[idx]

        active_observed = np.zeros(1)
        active_observed[0] = target_observed[idx]

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        target_sampler = mv.setup_target(active_target, active_observed)

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=burnin, ndraw=ndraw) # twosided by default
        pvalues.append(pval)

        # true selected

        def active_target(Y_star):
            result = boot_target(Y_star)
            return np.hstack([result[idx], result[nactive:]])

        active_observed = np.zeros_like(active_target(np.random.standard_normal(n)))
        active_observed[0] = target_observed[idx]
        active_observed[1:] = target_observed[nactive:]

        target_sampler = mv.setup_target(active_target, active_observed, target_set=[0])

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, active_observed, burnin=burnin, ndraw=ndraw) # twosided by default
        pvalues.append(pval)

        return pvalues
