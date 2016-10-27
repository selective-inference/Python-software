from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.decorators import (wait_for_return_value, 
                                        set_seed_iftrue, 
                                        set_sampling_params_iftrue,
                                        register_report)
from selection.tests.instance import logistic_instance

from selection.randomized.api import (randomization, 
                                      multiple_queries, 
                                      glm_group_lasso,
                                      glm_nonparametric_bootstrap,
                                      pairs_bootstrap_glm)
try:
    import statsmodels.api as sm
    statsmodels_available = True
except ImportError:
    statsmodels_available = False

instance_opts = {'snr':15,
                 's':5,
                 'p':20,
                 'n':200,
                 'rho':0.1}

def generate_data(s=5, 
                  n=200, 
                  p=20, 
                  rho=0.1, 
                  snr=15):

    return logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr, scale=False, center=False)

DEBUG = False
@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=100, burnin=100)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_scaling(snr=15, 
                 s=5, 
                 n=200, 
                 p=20, 
                 rho=0.1, 
                 burnin=20000, 
                 ndraw=30000, 
                 scale=0.9,
                 nsim=None, # needed for decorator
                 frac=0.5): # 0.9 has roughly same screening probability as 50% data splitting, i.e. around 10%

    randomizer = randomization.laplace((p,), scale=scale)
    X, y, beta, _ = generate_data(n=n, p=p, s=s, rho=rho, snr=snr)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1. / np.sqrt(n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_queries([M_est])
    mv.solve()

    active = M_est.selection_variable['variables']
    nactive = active.sum()

    if set(nonzero).issubset(np.nonzero(active)[0]):

        pvalues = []
        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        active_selected = A = [i for i in np.arange(active_set.shape[0]) if active_set[i] in nonzero]

        if not I:
            return None
        idx = I[0]
        inactive = ~M_est.selection_variable['variables']
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=inactive)

        if DEBUG:
            sampler = lambda : np.random.choice(n, size=(n,), replace=True)
            print(boot_target(sampler())[-3:], 'boot target')

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        # null saturated

        def null_target(indices):
            result = boot_target(indices)
            return result[idx]

        null_observed = np.zeros(1)
        null_observed[0] = target_observed[idx]

        target_sampler = mv.setup_target(null_target, null_observed)

        #target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2# should have something do with noise scale too

        print(target_sampler.crude_lipschitz(), 'crude')

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, 
                                              test_stat(null_observed), 
                                              burnin=burnin, 
                                              ndraw=ndraw, 
                                              stepsize=.5/target_sampler.crude_lipschitz()) # twosided by default
        pvalues.append(pval)

        # true saturated

        idx = A[0]

        def active_target(indices):
            result = boot_target(indices)
            return result[idx]

        active_observed = np.zeros(1)
        active_observed[0] = target_observed[idx]

        target_sampler = mv.setup_target(active_target, active_observed)
        target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2# should have something do with noise scale too

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, 
                                              test_stat(active_observed), 
                                              burnin=burnin, 
                                              ndraw=ndraw, 
                                              stepsize=.5/target_sampler.crude_lipschitz()) # twosided by default
        pvalues.append(pval)

        # null selected

        idx = I[0]

        def null_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        null_observed = np.zeros_like(null_target(range(n)))
        null_observed[0] = target_observed[idx]
        null_observed[1:] = target_observed[nactive:] 

        target_sampler = mv.setup_target(null_target, null_observed)#, target_set=[0])
        target_scaling = 5 * np.linalg.svd(target_sampler.target_transform[0][0])[1].max()**2# should have something do with noise scale too

        print(target_sampler.crude_lipschitz(), 'crude')

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, 
                                              test_stat(null_observed), 
                                              burnin=burnin, 
                                              ndraw=ndraw, 
                                              stepsize=.5/target_sampler.crude_lipschitz()) # twosided by default
        pvalues.append(pval)

        # true selected

        idx = A[0]

        def active_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        active_observed = np.zeros_like(active_target(range(n)))
        active_observed[0] = target_observed[idx] 
        active_observed[1:] = target_observed[nactive:]

        target_sampler = mv.setup_target(active_target, active_observed)#, target_set=[0])

        test_stat = lambda x: x[0]
        pval = target_sampler.hypothesis_test(test_stat, 
                                              test_stat(active_observed), 
                                              burnin=burnin, 
                                              ndraw=ndraw, 
                                              stepsize=.5/target_sampler.crude_lipschitz()) # twosided by default
        pvalues.append(pval)

        # condition on opt variables

        ### NOT WORKING -- need to implement conditioning within M_estimator!!!

        if False:

            # null saturated

            idx = I[0]

            def null_target(indices):
                result = boot_target(indices)
                return result[idx]

            null_observed = np.zeros(1)
            null_observed[0] = target_observed[idx]

            target_sampler = mv.setup_target(null_target, null_observed)

            print(target_sampler.crude_lipschitz(), 'crude')

            test_stat = lambda x: x[0]
            pval = target_sampler.hypothesis_test(test_stat, 
                                                  test_stat(null_observed), 
                                                  burnin=burnin, 
                                                  ndraw=ndraw, 
                                                  stepsize=.5/target_sampler.crude_lipschitz()) # twosided by default
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

            test_stat = lambda x: x[0]
            pval = target_sampler.hypothesis_test(test_stat, 
                                                  test_stat(active_observed), 
                                                  burnin=burnin, 
                                                  ndraw=ndraw, 
                                                  stepsize=.5/target_sampler.crude_lipschitz()) # twosided by default
            pvalues.append(pval)

        # true selected

        # oracle p-value -- draws a new data set

        X, y, beta, _ = generate_data(n=n, p=p, s=s, rho=rho, snr=snr)
        X_E = X[:,active_set]

        active_var = [False, True, False, True]

        if statsmodels_available:
            try:
                model = sm.GLM(y, X_E, family=sm.families.Binomial())
                model_results = model.fit()
                pvalues.extend([model_results.pvalues[I[0]], model_results.pvalues[A[0]]])
                active_var.extend([False, True])
            except sm.tools.sm_exceptions.PerfectSeparationError:
                pass
        else:
            pass

        # data splitting-ish p-value -- draws a new data set of smaller size
        # frac is presumed to be how much data was used in stage 1, we get (1-frac)*n for stage 2
        # frac defaults to 0.5

        Xs, ys, beta, _ = generate_data(n=n, p=p, s=s, rho=rho, snr=snr)
        Xs = Xs[:int((1-frac)*n)]
        ys = ys[:int((1-frac)*n)]
        X_Es = Xs[:,active_set]


        if statsmodels_available:
            try:
                model = sm.GLM(ys, X_Es, family=sm.families.Binomial())
                model_results = model.fit()
                pvalues.extend([model_results.pvalues[I[0]], model_results.pvalues[A[0]]])
                active_var.extend([False, False])
            except sm.tools.sm_exceptions.PerfectSeparationError:
                pass
        else:
            pass

        return pvalues, active_var

def data_splitting_screening(frac=0.5, snr=15, s=5, n=200, p=20, rho=0.1):

    count = 0
    
    while True:
        count += 1
        X, y, beta, _ = generate_data(n=n, p=p, s=s, rho=rho, snr=snr)

        n2 = int(frac * n)
        X = X[:n2]
        y = y[:n2]

        nonzero = np.where(beta)[0]
        lam_frac = 1.

        loss = rr.glm.logistic(X, y)
        epsilon = 1. / np.sqrt(n2)

        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n2, 10000)))).max(0))
        W = np.ones(p)*lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        problem = rr.simple_problem(loss, penalty)
        quadratic = rr.identity_quadratic(epsilon, 0, 0, 0)

        soln = problem.solve(quadratic)
        active_set = np.nonzero(soln != 0)[0]
        if set(nonzero).issubset(active_set):
            return count

def randomization_screening(scale=1., snr=15, s=5, n=200, p=20, rho=0.1):

    count = 0

    randomizer = randomization.laplace((p,), scale=scale)

    while True:
        count += 1
        X, y, beta, _ = generate_data(n=n, p=p, s=s, rho=rho, snr=snr)

        nonzero = np.where(beta)[0]
        lam_frac = 1.

        loss = rr.glm.logistic(X, y)
        epsilon = 1. / np.sqrt(n)

        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
        W = np.ones(p)*lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)

        M_est = glm_group_lasso(loss, epsilon, penalty, randomizer)
        M_est.solve()

        active_set = np.nonzero(M_est.initial_soln != 0)[0]
        if set(nonzero).issubset(active_set):
            return count
        
def report(niter=50, **kwargs):
    # these are all our null tests
    fn_names = ['test_scaling']

    dfs = []
    for fn in fn_names:
        fn = reports.reports[fn]
        dfs.append(reports.collect_multiple_runs(fn['test'],
                                                 fn['columns'],
                                                 niter,
                                                 reports.summarize_all))
    dfs = pd.concat(dfs)

    fig = reports.pvalue_plot(dfs, colors=['r', 'g'])

    fig.savefig('scaling_pvalues.pdf') # will have both bootstrap and CLT on plot
