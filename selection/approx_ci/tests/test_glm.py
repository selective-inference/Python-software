from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
import selection.tests.reports as reports
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_via_approx_density import approximate_conditional_density
from selection.approx_ci.estimator_approx import M_estimator_approx

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues


@register_report(['cover', 'ci_length', 'truth', 'naive_cover', 'naive_pvalues'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_approximate_ci(n=100,
                        p=10,
                        s=3,
                        snr=5,
                        rho=0.1,
                        lam_frac = 1.,
                        loss='gaussian',
                        randomizer='gaussian'):

    from selection.api import randomization

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    if randomizer=='gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer=='laplace':
        randomization = randomization.laplace((p,), scale=1.)

    M_est = M_estimator_approx(loss, epsilon, penalty, randomization, randomizer)
    M_est.solve_approx()
    ci = approximate_conditional_density(M_est)
    ci.solve_approx()

    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])

    true_support = np.asarray([i for i in range(p) if i < s])

    nactive = np.sum(active)

    print("active set, true_support", active_set, true_support)

    true_vec = beta[active]

    print("true coefficients", true_vec)

    if (set(active_set).intersection(set(true_support)) == set(true_support))== True:

        ci_active = np.zeros((nactive, 2))
        #mle_active = np.zeros(nactive)
        covered = np.zeros(nactive, np.bool)
        ci_length = np.zeros(nactive)
        pivots = np.zeros(nactive)

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape
        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_pvals = naive_pvalues(target, M_est.target_observed, true_vec)
        naive_covered = np.zeros(nactive)
        toc = time.time()

        for j in range(nactive):
            ci_active[j, :] = np.array(ci.approximate_ci(j))
            #mle_active[j] = ci.approx_MLE_solver(j, nstep= 100)[0]
            if (ci_active[j, 0] <= true_vec[j]) and (ci_active[j,1] >= true_vec[j]):
                covered[j] = 1
            ci_length[j] = ci_active[j,1] - ci_active[j,0]
            print(ci_active[j, :])
            pivots[j] = ci.approximate_pvalue(j, true_vec[j])

            # naive ci
            if (ci_naive[j,0]<=true_vec[j]) and (ci_naive[j,1]>=true_vec[j]):
                naive_covered[j]+=1

        tic = time.time()
        print('ci time now', tic - toc)
        return covered, ci_length, pivots, naive_covered, naive_pvals


def report(niter=50, **kwargs):

    kwargs = {'s': 0, 'n': 200, 'p': 30, 'snr': 7, 'loss': 'gaussian', 'randomizer':'gaussian'}
    split_report = reports.reports['test_approximate_ci']
    screened_results = reports.collect_multiple_runs(split_report['test'],
                                                     split_report['columns'],
                                                     niter,
                                                     reports.summarize_all,
                                                     **kwargs)

    fig = reports.pivot_plot_plus_naive(screened_results)
    fig.savefig('approx_pivots_glm.pdf')


if __name__=='__main__':
    report()