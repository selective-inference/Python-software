import numpy as np
from scipy.stats import norm as ndist
import pandas as pd
import regreg.api as rr
import selection.api as sel
from selection.tests.instance import (gaussian_instance, logistic_instance)
from selection.randomized.glm import (pairs_bootstrap_glm,
                                      glm_nonparametric_bootstrap)
from selection.algorithms.lasso import (glm_sandwich_estimator,
                                        lasso)
from selection.constraints.affine import (constraints,
                                          stack)
from selection.randomized.cv_view import CV_view
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
from selection.randomized.tests.test_cv_lee_et_al import pivot, equal_tailed_interval

@register_report(['pvalue', 'cover', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive',
                  'active_var'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_cv_corrected_nonrandomized_lasso(n=3000,
                                    p=1000,
                                    s=10,
                                    snr = 3.5,
                                    rho = 0.,
                                    sigma = 1.,
                                    K = 5,
                                    loss="gaussian",
                                    X = None,
                                    check_screen=True,
                                    intervals=False):

    print (n, p, s, rho)
    if X is not None:
        beta = np.zeros(p)
        beta[:s] = snr

    if loss == "gaussian":
        if X is None:
            X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1, scale=True, center=True)
        else:
            y = X.dot(beta) + np.random.standard_normal(n)*sigma
        glm_loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        if X is None:
            X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr, scale=True, center=True)
        else:
            eta = np.dot(X, beta)
            pi = np.exp(eta) / (1 + np.exp(eta))
            y = np.random.binomial(1, pi)
        glm_loss = rr.glm.logistic(X, y)

    truth = np.nonzero(beta != 0)[0]

    cv = CV_view(glm_loss, loss_label=loss, lasso_randomization=None, epsilon=None,
                 scale1=0.01, scale2=0.01)
    # views.append(cv)
    cv.solve(glmnet=True)
    lam_CV_randomized = cv.lam_CVR
    print("minimizer of CVR", lam_CV_randomized)

    CV_val_randomized = cv.observed_opt_state
    CV_boot = cv.CVR_boot

    # Implemented the post-selection p-values for  the randomized CV
    L = lasso.gaussian(X, y, lam_CV_randomized)
    L.covariance_estimator = glm_sandwich_estimator(L.loglike, B=2000)
    soln = L.fit()


    active = soln !=0
    nactive = active.sum()
    print("nactive", nactive)
    if nactive==0:
        return None
    active_signs = soln[active]

    selected_boot = pairs_bootstrap_glm(L.loglike,active)[0]

    # this is the \beta_E computed at the bootstrapped X and y.
    def coef_boot(indices):
        # bootstrap of just coefficients
        return selected_boot(indices)[:active.sum()]


    if (check_screen==False) or (set(truth).issubset(np.nonzero(active)[0])):

        active_set = np.nonzero(active)[0]
        true_vec = beta[active]
        one_step = L.onestep_estimator

        cov_est = glm_nonparametric_bootstrap(n, n)
        # compute covariance of selected parameters with CV error curve
        cov = cov_est(coef_boot, cross_terms=[CV_boot], nsample=500)

        # residual is fixed
        # covariance of L.constraints is more accurate than cov[0]
        # but estimates the same thing (i.e. more bootstrap replicates)
        A = cov[1].T.dot(np.linalg.pinv(L.constraints.covariance))
        residual = CV_val_randomized- A.dot(one_step)

        # minimizer indicator

        lam_idx_randomized =  cv.lam_idx #list(CV_view.lam_seq).index(lam_CV_randomized)
        lam_keep_randomized = np.zeros(CV_val_randomized.shape[0], np.bool)
        lam_keep_randomized[lam_idx_randomized] = 1
        B = -np.identity(CV_val_randomized.shape[0])
        B += (np.multiply.outer(lam_keep_randomized, np.ones_like(lam_keep_randomized))).T ## and here

        keep = np.ones(CV_val_randomized.shape[0], np.bool)
        keep[lam_idx_randomized] = 0
        B = B[keep]
        C = B.dot(A)

        CV_constraints = constraints(C, -B.dot(residual))

        full_constraints = stack(CV_constraints, L.constraints)
        full_constraints.covariance[:] = L.constraints.covariance

        # CV corrected

        pvalues = np.zeros(nactive)
        sel_length = np.zeros(nactive)
        sel_covered = np.zeros(nactive)
        active_var = np.zeros(nactive, np.bool)

        naive_pvalues = np.zeros(nactive)
        naive_length = np.zeros(nactive)
        naive_covered = np.zeros(nactive)

        #if not full_constraints(one_step):
        #    raise ValueError('constraints are wrong')

        C = full_constraints
        if C is not None:
            one_step = L.onestep_estimator
            for i in range(one_step.shape[0]):
                eta = np.zeros_like(one_step)
                eta[i] = active_signs[i]
                alpha = 0.1
                if C.linear_part.shape[0] > 0:  # there were some constraints
                    L, Z, U, S = C.bounds(eta, one_step)
                    _pval = pivot(L, Z, U, S)
                    # two-sided
                    _pval = 2 * min(_pval, 1 - _pval)
                    if intervals==True:
                        if _pval < 10 ** (-8):
                            return None
                        L, Z, U, S = C.bounds(eta, one_step)
                        _interval = equal_tailed_interval(L, Z, U, S, alpha=alpha)
                        _interval = sorted([_interval[0] * active_signs[i],
                                        _interval[1] * active_signs[i]])
                else:
                    obs = (eta * one_step).sum()
                    sd = np.sqrt((eta * C.covariance.dot(eta)))
                    Z = obs / sd
                    # use Phi truncated to [-5,5]
                    _pval = 2 * (ndist.sf(min(np.fabs(Z))) - ndist.sf(5)) / (ndist.cdf(5) - ndist.cdf(-5))
                    if intervals==True:
                        _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                                 obs + ndist.ppf(1 - alpha / 2) * sd)
                pvalues[i] = _pval

                def naive_inference():
                    obs = (eta * one_step).sum()
                    sd = np.sqrt(np.dot(eta.T, C.covariance.dot(eta)))
                    Z = obs / sd
                    # use Phi truncated to [-5,5]
                    _pval = ndist.cdf(obs / sigma)
                    _pval = 2 * min(_pval, 1 - _pval)
                    _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                                 obs + ndist.ppf(1 - alpha / 2) * sd)
                    return _pval, _interval

                naive_pvalues[i], _naive_interval = naive_inference()

                # print(_naive_interval)

                def coverage(LU):
                    L, U = LU[0], LU[1]
                    _length = U - L
                    _covered = 0
                    if (L <= true_vec[i]) and (U >= true_vec[i]):
                        _covered = 1
                    return _covered, _length

                if intervals == True:
                    sel_covered[i], sel_length[i] = coverage(_interval)
                    naive_covered[i], naive_length[i] = coverage(_naive_interval)
                active_var[i] = active_set[i] in truth

        print(pvalues)
        return pvalues, sel_covered, sel_length, \
               naive_pvalues, naive_covered, naive_length, active_var


def report(niter=100, design="random", **kwargs):

    if design == "fixed":
        X, _, _, _, _ = gaussian_instance(**kwargs)
        kwargs.update({'X': X})

    intervals_report = reports.reports['test_cv_corrected_nonrandomized_lasso']
    screened_results = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)
    screened_results.to_pickle("cv_corrected_nonrandomized_lasso.pkl")
    results = pd.read_pickle("cv_corrected_nonrandomized_lasso.pkl")

    fig = reports.pvalue_plot(results, label = 'CV corrected')
    fig.suptitle("CV corrected norandomized Lasso pivots", fontsize=20)
    fig.savefig('cv_corrected_nonrandomized_lasso_pivots.pdf')


if __name__ == '__main__':
    np.random.seed(500)
    kwargs = {'s': 0, 'n': 500, 'p': 100, 'snr': 3.5, 'sigma': 1, 'rho': 0., 'intervals':False}
    report(niter=1, **kwargs)