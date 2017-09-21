from __future__ import print_function
import numpy as np
from selection.tests.instance import gaussian_instance,logistic_instance
import regreg.api as rr

from selection.randomized.M_estimator_group_lasso import restricted_Mest
from selection.randomized.M_estimator_group_lasso import M_estimator
import selection.tests.reports as reports
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
from selection.tests.flags import SMALL_SAMPLES, SET_SEED

@register_report(['pivot', 'covered_clt'])
@wait_for_return_value()
def test_nonrandomized(s=0,
                       n=200,
                       p=10,
                       signal=7,
                       rho=0,
                       lam_frac=0.8,
                       loss='gaussian',
                       solve_args={'min_its': 20, 'tol': 1.e-10}):
    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    nonzero = np.where(beta)[0]
    print("lam", lam)
    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    true_vec = beta
    M_est = M_estimator(lam, loss, penalty)
    M_est.solve()
    active  = M_est._overall
    nactive = np.sum(active)
    print("nactive", nactive)
    if nactive == 0:
        return None

    #score_mean = M_est.observed_internal_state.copy()
    #score_mean[nactive:] = 0
    M_est.setup_sampler(score_mean = np.zeros(p))
    #M_est.setup_sampler(score_mean=score_mean)
    #M_est.sample(ndraw = 1000, burnin=1000, stepsize=1./p)

    if set(nonzero).issubset(np.nonzero(active)[0]):
        check_screen=True
        #test_stat = lambda x: np.linalg.norm(x)
        #return M_est.hypothesis_test(test_stat, test_stat(M_est.observed_internal_state), stepsize=1./p)

        ci = M_est.confidence_intervals(M_est.observed_internal_state)
        pivots = M_est.coefficient_pvalues(M_est.observed_internal_state)
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
        covered = coverage(ci)[0]
        #print(pivots)
        #print(coverage)
        return pivots, covered

def report(niter=100, **kwargs):

    kwargs = {'s': 0, 'n': 300, 'p': 10, 'signal': 7}
    split_report = reports.reports['test_nonrandomized']
    screened_results = reports.collect_multiple_runs(split_report['test'],
                                                     split_report['columns'],
                                                     niter,
                                                     reports.summarize_all,
                                                     **kwargs)

    fig = reports.pivot_plot_simple(screened_results)
    fig.savefig('nonrandomized_pivots.pdf') # will have both bootstrap and CLT on plot


def main():
    report()
