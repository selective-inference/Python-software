import functools

import numpy as np
from scipy.stats import norm as ndist
import matplotlib.pyplot as plt

import regreg.api as rr
from selection.api import (randomization,
                           glm_group_lasso,
                           multiple_queries,
                           glm_target)
import selection.api as sel
from selection.tests.instance import gaussian_instance
from selection.randomized.glm import (pairs_bootstrap_glm,
                                      glm_nonparametric_bootstrap)
from selection.algorithms.lasso import (glm_sandwich_estimator,
                                        lasso)
from selection.constraints.affine import (constraints,
                                          stack)
from selection.randomized.query import naive_confidence_intervals

import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report


def CV_err(X, y, lam, folds, scale=0.5):

    n, p = X.shape

    penalty = rr.l1norm(p, lagrange=lam)

    CV_err = 0
    CV_err_randomized = 0

    CV_err_squared = 0
    CV_err_squared_randomized = 0

    for fold in np.unique(folds):
        test = folds == fold
        train = ~test

        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        n_test = y_test.shape[0]

        loss = rr.glm.gaussian(X_train, y_train)
        problem = rr.simple_problem(loss, penalty)
        beta_train = problem.solve()

        resid = y_test - X_test.dot(beta_train)
        cur = (resid**2).sum() / n_test

        # there are several ways we could randomize here...
        random_noise = scale * np.random.standard_normal(y_test.shape)
        cur_randomized = ((resid + random_noise)**2).sum() / n_test

        CV_err += cur
        CV_err_squared += cur**2

        CV_err_randomized += cur_randomized
        CV_err_squared_randomized += cur_randomized**2

    K = len(np.unique(folds))

    SD_CV = np.sqrt((CV_err_squared.mean() - CV_err.mean()**2) / (K-1))
    SD_CV_randomized = np.sqrt((CV_err_squared_randomized.mean() - CV_err_randomized.mean()**2) / (K-1))
    return CV_err, SD_CV, CV_err_randomized, SD_CV_randomized

def choose_lambda_CV(X, y, lam_seq, folds):

    CV_curve = []
    for lam in lam_seq:
        CV_curve.append(CV_err(X, y, lam, folds) + (lam,))

    CV_curve = np.array(CV_curve)
    minCV = lam_seq[np.argmin(CV_curve[:,0])] # unrandomized
    minCV_randomized = lam_seq[np.argmin(CV_curve[:,2])] # randomized

    return minCV_randomized, CV_curve

def bootstrap_CV_curve(X, y, lam_seq, folds, K):

    def _bootstrap_CVerr_curve(X, y, lam_seq, K, indices):
        n, p = X.shape
        folds_star = np.arange(n) % K
        np.random.shuffle(folds_star)
        X_star, y_star = X[indices], y[indices]
        return np.array(choose_lambda_CV(X_star, y_star, lam_seq, folds_star)[1])[:,0]

    return functools.partial(_bootstrap_CVerr_curve, X, y, lam_seq, K)

@register_report(['mle', 'truth', 'pvalue', 'cover', 'naive_cover', 'active'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_cv(n=100, p=20, s=10, snr=5, K=5, rho=0,
             randomizer='laplace',
             intervals = 'old',
             bootstrap=False,
             ndraw = 10000,
             burnin = 2000):
    print (n, p, s, rho)
#     X, y, _, truth, sigma = gaussian_instance(n=n,
#                                               p=p,
#                                               s=s,
#                                               rho=rho)
    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=1.)
    elif randomizer == 'gaussian':
        randomizer = randomization.gaussian(np.identity(p))
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=1.)

    #X = np.random.standard_normal((n, p))
    #y = np.random.standard_normal(n)
    #truth = np.array([], np.int)
    #sigma = 1.
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
    truth = beta
    loss = rr.glm.gaussian(X, y)
    active = np.nonzero(truth != 0)[0]
    lam_seq = np.exp(np.linspace(np.log(1.e-2), np.log(1), 30)) * np.fabs(X.T.dot(y)).max()

    folds = np.arange(n) % K
    np.random.shuffle(folds)

    lam_CV, CV_curve = choose_lambda_CV(X, y, lam_seq, folds)
    CV_val = np.array(CV_curve)[:,2]

    #L = lasso.gaussian(X, y, lam_CV)
    #L.covariance_estimator = glm_sandwich_estimator(L.loglike, B=2000)
    #soln = L.fit()

    W = np.ones(p) * lam_CV
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    epsilon = 1.
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_queries([M_est1])
    mv.solve()

    CV_boot = bootstrap_CV_curve(X, y, lam_seq, folds, K)

    #active = soln != 0
    active_union = M_est1._overall
    nactive = np.sum(active_union)
    print(nactive)

    _full_boot_score = pairs_bootstrap_glm(loss,
                                           active_union)[0]
    def _boot_score(indices):
        return _full_boot_score(indices)[:nactive]

    cov_est = glm_nonparametric_bootstrap(n, n)
    # compute covariance of selected parameters with CV error curve
    cov = cov_est(CV_boot, cross_terms=[_boot_score], nsample=1)
    if bootstrap=='False':
        M_est1.target_decomposition(cov, CV_val)


    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]
        true_vec = beta[active_union]

        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     mv,
                                                     bootstrap=bootstrap)

        if intervals == 'old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            LU = target_sampler.confidence_intervals(target_observed,
                                                     sample=target_sample,
                                                     level=0.9)
            pivots_mle = target_sampler.coefficient_pvalues(target_observed,
                                                            parameter=target_sampler.reference,
                                                            sample=target_sample)
            pivots_truth = target_sampler.coefficient_pvalues(target_observed,
                                                              parameter=true_vec,
                                                              sample=target_sample)
            pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                         parameter=np.zeros_like(true_vec),
                                                         sample=target_sample)
        else:
            full_sample = target_sampler.sample(ndraw=ndraw,
                                                burnin=burnin,
                                                keep_opt=True)
            LU = target_sampler.confidence_intervals_translate(target_observed,
                                                               sample=full_sample,
                                                               level=0.9)
            pivots_mle = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                      parameter=target_sampler.reference,
                                                                      sample=full_sample)
            pivots_truth = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                        parameter=true_vec,
                                                                        sample=full_sample)
            pvalues = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                   parameter=np.zeros_like(true_vec),
                                                                   sample=full_sample)

        LU_naive = naive_confidence_intervals(target_sampler, target_observed)

        L, U = LU.T

        covered = np.zeros(nactive, np.bool)
        naive_covered = np.zeros(nactive, np.bool)
        active_var = np.zeros(nactive, np.bool)

        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            if (LU_naive[j, 0] <= true_vec[j]) and (LU_naive[j, 1] >= true_vec[j]):
                naive_covered[j] = 1
            active_var[j] = active_set[j] in nonzero

        return pivots_mle, pivots_truth, pvalues, covered, naive_covered, active_var


def report(niter=5, **kwargs):
    kwargs = {'s': 0, 'n': 200, 'p': 10, 'snr': 7, 'bootstrap': False, 'randomizer': 'gaussian'}
    intervals_report = reports.reports['test_cv']
    CLT_runs = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    # fig = reports.pivot_plot(CLT_runs, color='b', label='CLT')
    fig = reports.pivot_plot_2in1(CLT_runs, color='b', label='CLT')

    #kwargs['bootstrap'] = True
    #bootstrap_runs = reports.collect_multiple_runs(intervals_report['test'],
    #                                               intervals_report['columns'],
    #                                               niter,
    #                                               reports.summarize_all,
    #                                               **kwargs)

    # fig = reports.pivot_plot(bootstrap_runs, color='g', label='Bootstrap', fig=fig)
    #fig = reports.pivot_plot_2in1(bootstrap_runs, color='g', label='Bootstrap', fig=fig)

    fig.savefig('intervals_pivots.pdf')  # will have both bootstrap and CLT on plot


if __name__ == '__main__':
    report()