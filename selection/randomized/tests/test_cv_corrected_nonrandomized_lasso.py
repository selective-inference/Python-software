import numpy as np
from scipy.stats import norm as ndist
import matplotlib.pyplot as plt

import regreg.api as rr

import selection.api as sel
from selection.tests.instance import (gaussian_instance, logistic_instance)
from selection.randomized.glm import (pairs_bootstrap_glm,
                                      glm_nonparametric_bootstrap)
from selection.algorithms.lasso import (glm_sandwich_estimator,
                                        lasso)
from selection.constraints.affine import (constraints,
                                          stack)
from selection.randomized.cv import CV
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report

@register_report(['pvalue', 'cover', 'ci_length_clt', 'active_var'])
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
                                    X = None):

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

    lam_seq = np.exp(np.linspace(np.log(1.e-6), np.log(2), 30)) * np.mean(np.fabs(np.dot(X.T, y)).max(0))
    folds = np.arange(n) % K
    np.random.shuffle(folds)

    CV_compute = CV(glm_loss, folds, lam_seq)
    _, _,_, lam_CV_randomized, CV_val_randomized, _ = CV_compute.choose_lambda_CVr(scale=1.)

    # Implemented the post-selection p-values for  the randomized CV
    L = lasso.gaussian(X, y, lam_CV_randomized)
    L.covariance_estimator = glm_sandwich_estimator(L.loglike, B=2000)
    soln = L.fit()

    _, CV_boot = CV_compute.bootstrap_CVr_curve()

    active = soln !=0
    nactive = active.sum()
    print("nactive", nactive)
    if nactive==0:
        return None

    selected_boot = pairs_bootstrap_glm(L.loglike,active)[0]

    # this is the \beta_E computed at the bootstrapped X and y.
    def coef_boot(indices):
        # bootstrap of just coefficients
        return selected_boot(indices)[:active.sum()]

    check_screen = True
    if check_screen == True:
        if set(truth).issubset(np.nonzero(active)[0]):
            check_screen = False

    if check_screen==False:

        active_set = np.nonzero(active)[0]
        true_vec = beta[active]
        one_step = L.onestep_estimator

        cov_est = glm_nonparametric_bootstrap(n, n)
        # compute covariance of selected parameters with CV error curve
        cov = cov_est(coef_boot, cross_terms=[CV_boot],nsample=10)

        # residual is fixed
        # covariance of L.constraints is more accurate than cov[0]
        # but estimates the same thing (i.e. more bootstrap replicates)
        A = cov[1].T.dot(np.linalg.pinv(L.constraints.covariance))
        residual = CV_val_randomized- A.dot(one_step)

        # minimizer indicator

        lam_idx_randomized = list(lam_seq).index(lam_CV_randomized)
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

        if not full_constraints(one_step):
            raise ValueError('constraints are wrong')

        for i in range(active.sum()):
            active_var[i] = active_set[i] in truth

            keep_randomized = np.zeros(active.sum())
            keep_randomized[i] = 1.

            pvalues[i] = full_constraints.pivot(keep_randomized,
                                                      one_step,
                                                      alternative='twosided')
            interval = full_constraints.interval(keep_randomized,
                                                one_step, alpha=0.1)
            sel_length[i] = interval[1] - interval[0]
            if (interval[0] <= true_vec[i]) and (interval[1] >= true_vec[i]):
                sel_covered[i] = 1

        return pvalues, sel_covered, sel_length, active_var


def report(niter=100, **kwargs):

    kwargs = {'s': 0, 'n': 100, 'p': 50, 'snr': 3.5, 'sigma':1, 'rho':0.}
    #X, _, _, _, _ = gaussian_instance(**kwargs)
    #kwargs.update({'X':X})
    intervals_report = reports.reports['test_cv_corrected_nonrandomized_lasso']
    CV_runs = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    fig = reports.pvalue_plot(CV_runs, label = 'CV corrected')
    fig.suptitle("CV corrected norandomized Lasso pivots")
    fig.savefig('cv_corrected_nonrandomized_lasso_pivots.pdf')


if __name__ == '__main__':
    np.random.seed(500)
    report()
    #compute_power()