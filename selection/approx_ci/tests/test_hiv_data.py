from __future__ import print_function
import os, numpy as np, pandas, statsmodels.api as sm
import regreg.api as rr
from selection.approx_ci.selection_map import M_estimator_map
from selection.approx_ci.ci_approx_density import approximate_conditional_density

from selection.randomized.query import naive_confidence_intervals

def hiv_inference_test():
    if not os.path.exists("NRTI_DATA.txt"):
        NRTI = pandas.read_table(
            "http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt", na_values="NA")
    else:
        NRTI = pandas.read_table("NRTI_DATA.txt")

    NRTI_specific = []
    NRTI_muts = []
    for i in range(1, 241):
        d = NRTI['P%d' % i]
        for mut in np.unique(d):
            if mut not in ['-', '.'] and len(mut) == 1:
                test = np.equal(d, mut)
                if test.sum() > 10:
                    NRTI_specific.append(np.array(np.equal(d, mut)))
                    NRTI_muts.append("P%d%s" % (i, mut))

    NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)

    X_NRTI = np.array(NRTI_specific, np.float)
    Y = NRTI['3TC']  # shorthand
    keep = ~np.isnan(Y).astype(np.bool)
    X_NRTI = X_NRTI[np.nonzero(keep)];
    Y = Y[keep]
    Y = np.array(np.log(Y), np.float);
    Y -= Y.mean()
    X_NRTI -= X_NRTI.mean(0)[None, :];
    X_NRTI /= X_NRTI.std(0)[None, :]
    X = X_NRTI  # shorthand
    n, p = X.shape
    X /= np.sqrt(n)

    ols_fit = sm.OLS(Y, X).fit()
    sigma_3TC = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1)

    lam_frac = 1.
    loss = rr.glm.gaussian(X, Y)
    epsilon = 1. / np.sqrt(n)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_3TC
    print(lam)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

    from selection.api import randomization
    randomization = randomization.isotropic_gaussian((p,), scale=1.)

    #change grid for parameter for HIV data
    M_est = M_estimator_map(loss, epsilon, penalty, randomization, randomization_scale=0.7)
    M_est.solve_approx()
    active = M_est._overall
    nactive = np.sum(active)

    ci_active = np.zeros((nactive, 2))
    ci_length = np.zeros(nactive)
    mle_active = np.zeros((nactive, 1))

    ci = approximate_conditional_density(M_est)
    ci.solve_approx()

    class target_class(object):
        def __init__(self, target_cov):
            self.target_cov = target_cov
            self.shape = target_cov.shape

    target = target_class(M_est.target_cov)
    ci_naive = naive_confidence_intervals(target, M_est.target_observed)

    for j in range(nactive):
        ci_active[j, :] = np.array(ci.approximate_ci(j))
        ci_length[j] = ci_active[j, 1] - ci_active[j, 0]
        mle_active[j, :] = ci.approx_MLE_solver(j, nstep=100)[0]

    unadjusted_mle = np.zeros((nactive, 1))
    for j in range(nactive):
        unadjusted_mle[j, :] = ci.target_observed[j]

    adjusted_intervals = np.hstack([mle_active, ci_active]).T
    unadjusted_intervals = np.hstack([unadjusted_mle, ci_naive]).T

    print("adjusted confidence", adjusted_intervals)
    print("naive confidence", unadjusted_intervals)

    intervals = np.vstack([unadjusted_intervals, adjusted_intervals])

    return intervals