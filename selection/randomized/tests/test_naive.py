import numpy as np
import regreg.api as rr
import pandas as pd
import selection.api as sel
from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import lasso
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
from statsmodels.sandbox.stats.multicomp import multipletests
from selection.randomized.cv_view import CV_view
from scipy.stats import norm as ndist
from scipy.optimize import bisect
from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)

def compute_projection_parameters(n, p, s, signal, rho, sigma, active):
    multiple = 10**2
    n_large = multiple*n
    X_large = np.zeros((n_large,p))
    y_large = np.zeros(n_large)

    for i in range(multiple):
        X_large[(i*n):((i+1)*n), :], y_large[(i*n):((i+1)*n)], _, _, _ = \
            gaussian_instance(n=n, p=p, s=s, signal=signal, rho=rho, sigma=sigma, scale=True, center=True)

    proj_param = np.linalg.lstsq(X_large[:, active], y_large)[0]
    print(proj_param)
    return proj_param




@register_report(['naive_pvalues', 'covered_naive', 'ci_length_naive', 'active_var'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_naive(n=300,
               p=100,
               s=10,
               signal = 3.5,
               rho = 0.,
               sigma = 1.,
               cross_validation=True,
               condition_on_CVR=False,
               lam_frac = 1.,
               X = None,
               check_screen = False,
               check_projection_param = False,
               check_selected_param = True,
               intervals = False):

    print(n, p, s)

    if X is None:
        X, y, beta, truth, sigma = gaussian_instance(n=n, p=p, s=s, signal=signal, rho=rho, \
                                                     sigma=sigma, scale=True, center=True)
    else:
        beta = np.zeros(p)
        beta[:s] = signal
        y = X.dot(beta) + np.random.standard_normal(n)*sigma

    truth = np.nonzero(beta != 0)[0]

    if cross_validation:
        cv = CV_view(rr.glm.gaussian(X,y), loss_label="gaussian", lasso_randomization=None, epsilon=None,
                     scale1=None, scale2=None)

        cv.solve(glmnet=True)
        lam = cv.lam_CVR

        if condition_on_CVR:
            cv.condition_on_opt_state()
            lam = cv.one_SD_rule(direction="up")
    else:
        lam = lam_frac*np.fabs(X.T.dot(np.random.normal(1, 1. / 2, (n, 1000)))).max()

    L = lasso.gaussian(X, y, lam, sigma=sigma)
    soln = L.fit()

    active = soln != 0
    nactive = active.sum()
    print("nactive", nactive)
    if nactive==0:
        return None

    active_signs = np.sign(soln[active])
    active_set = np.nonzero(active)[0]


    if (check_screen==False):
        if check_projection_param==True:
            true_vec = compute_projection_parameters(n, p, s, signal, rho, sigma, active)
        else:
            true_vec = signal*np.array([active_set[i] in truth for i in range(nactive)], int)
        print(true_vec)
    else:
        true_vec = beta[active]


    if (check_screen == False) or (set(truth).issubset(np.nonzero(active)[0])):

        print("active set", active_set)
        active_var = np.zeros(nactive, np.bool)

        naive_pvalues = np.zeros(nactive)
        naive_length = np.zeros(nactive)
        naive_covered = np.zeros(nactive)

        C = L.constraints

        if C is not None:
            one_step = L.onestep_estimator
            for i in range(one_step.shape[0]):
                eta = np.zeros_like(one_step)
                eta[i] = active_signs[i]
                alpha = 0.1

                def naive_inference():
                    obs = (eta * one_step).sum()
                    sd = np.sqrt(np.dot(eta.T, C.covariance.dot(eta)))
                    Z = obs / sd
                    # use Phi truncated to [-5,5]
                    _pval = ndist.cdf(Z)
                    _pval = 2 * min(_pval, 1 - _pval)
                    _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                                 obs + ndist.ppf(1 - alpha / 2) * sd)
                    return _pval, _interval

                naive_pvalues[i], _naive_interval = naive_inference()

                def coverage(LU):
                    L, U = LU[0], LU[1]
                    _length = U - L
                    _covered = 0
                    if (L <= true_vec[i]) and (U >= true_vec[i]):
                        _covered = 1
                    return _covered, _length

                naive_covered[i], naive_length[i] = coverage(_naive_interval)
                active_var[i] = active_set[i] in truth
        else:
            return None

        print("naive pvalues",naive_pvalues)

        return  naive_pvalues, naive_covered, naive_length, active_var


def report(niter=50, design="random", **kwargs):

    if design=="fixed":
        X, _, _, _, _ = gaussian_instance(**kwargs)
        kwargs.update({'X':X})

    kwargs.update({'cross_validation':True, 'condition_on_CVR':False})
    intervals_report = reports.reports['test_naive']
    screened_results = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    screened_results.to_pickle("naive.pkl")
    results = pd.read_pickle("naive.pkl")

    fig = reports.naive_pvalue_plot(results)
    #fig = reports.pvalue_plot(results, label="Naive p-values")
    fig.suptitle("Naive p-values", fontsize=20)
    fig.savefig('naive_pvalues.pdf')


if __name__ == '__main__':

    np.random.seed(500)
    kwargs = {'s': 0, 'n': 100, 'p': 50, 'signal': 3.5, 'sigma': 1, 'rho': 0., 'intervals':True}
    report(niter=100, **kwargs)


