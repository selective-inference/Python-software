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

def test_cv_nonrandomized_lasso(n=3000,
                                p=1000,
                                s=10,
                                snr = 3.5,
                                rho =0.,
                                K = 5,
                                loss="gaussian"):

    print (n, p, s, rho)

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1, scale=True, center=True)
        glm_loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr, scale=True, center=True)
        glm_loss = rr.glm.logistic(X, y)

    truth = np.nonzero(beta != 0)[0]

    lam_seq = np.exp(np.linspace(np.log(1.e-6), np.log(2), 30)) * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))
    folds = np.arange(n) % K
    np.random.shuffle(folds)

    CV_compute = CV(glm_loss, folds, lam_seq)
    _, _, lam_CV_randomized, CV_val_randomized = CV_compute.choose_lambda_CVr(scale=1.)

    # Implemented the post-selection p-values for  the randomized CV
    L_randomized = lasso.gaussian(X, y, lam_CV_randomized)
    L_randomized.covariance_estimator = glm_sandwich_estimator(L_randomized.loglike, B=2000)
    soln_randomized = L_randomized.fit()

    CV_boot = CV_compute.bootstrap_CVr_curve()

    active_randomized = soln_randomized !=0

    selected_boot_randomized = pairs_bootstrap_glm(L_randomized.loglike,
                                        active_randomized)[0]

    # this is the \beta_E computed at the bootstrapped X and y.
    def coef_boot_randomized(indices):
        # bootstrap of just coefficients
        return selected_boot_randomized(indices)[:active_randomized.sum()]


    if set(truth).issubset(np.nonzero(active_randomized)[0]) and active_randomized.sum() > 0:

        print("nactive randomized", active_randomized.sum())
        active_set_randomized = np.nonzero(active_randomized)[0]

        cov_est = glm_nonparametric_bootstrap(n, n)
        # compute covariance of selected parameters with CV error curve
        cov_randomized = cov_est(coef_boot_randomized, cross_terms=[CV_boot],nsample=10)

        # residual is fixed
        # covariance of L.constraints is more accurate than cov[0]
        # but estimates the same thing (i.e. more bootstrap replicates)
        A_randomized = cov_randomized[1].T.dot(np.linalg.pinv(L_randomized.constraints.covariance))
        residual_randomized = CV_val_randomized- A_randomized.dot(L_randomized.onestep_estimator)

        # minimizer indicator

        lam_idx_randomized = list(lam_seq).index(lam_CV_randomized)
        lam_keep_randomized = np.zeros(CV_val_randomized.shape[0], np.bool)
        lam_keep_randomized[lam_idx_randomized] = 1
        B_randomized = -np.identity(CV_val_randomized.shape[0])
        B_randomized += (np.multiply.outer(lam_keep_randomized, np.ones_like(lam_keep_randomized))).T ## and here

        keep_randomized = np.ones(CV_val_randomized.shape[0], np.bool)
        keep_randomized[lam_idx_randomized] = 0
        B_randomized = B_randomized[keep_randomized]

        C_randomized = B_randomized.dot(A_randomized)

        CV_constraints_randomized = constraints(C_randomized,
                                     -B_randomized.dot(residual_randomized))

        full_constraints_randomized = stack(CV_constraints_randomized,
                                 L_randomized.constraints)
        full_constraints_randomized.covariance[:] = L_randomized.constraints.covariance

        # CV corrected

        P0_randomized, PA_randomized = [], []

        if not full_constraints_randomized(L_randomized.onestep_estimator):
            raise ValueError('constraints are wrong')

        for i in range(active_randomized.sum()):

            keep_randomized = np.zeros(active_randomized.sum())
            keep_randomized[i] = 1.

            pivot = full_constraints_randomized.pivot(keep_randomized,
                                           L_randomized.onestep_estimator,
                                           alternative='twosided')

            if active_set_randomized[i] in truth:
                PA_randomized.append(pivot)
            else:
                P0_randomized.append(pivot)

        return P0_randomized, PA_randomized
    else:
        return [],[]


if __name__ == "__main__":
    np.random.seed(500)
    P0_randomized, PA_randomized = [], []

    for i in range(200):
        print("iteration", i)
        p0_randomized, pA_randomized = test_cv_nonrandomized_lasso(n=50, p=20, s=0, snr=3.5, K=5, rho=0.)
        P0_randomized.extend(p0_randomized); PA_randomized.extend(pA_randomized)
        print(np.mean(P0_randomized), np.std(P0_randomized), np.mean(np.array(P0_randomized)<0.05),'null corrected with randomization')

        if len(P0_randomized) > 0:
            import statsmodels.api as sm
            import matplotlib.pyplot as plt
            U = np.linspace(0,1,101)
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P0_randomized)(U), label='corrected with randomization')
            plt.plot([0,1],[0,1], 'k--')
            plt.legend(loc='lower right')
            plt.xlabel("observed p-value")
            plt.ylabel("CDF")
            plt.suptitle("P-values using randomized CV error curve")
            plt.savefig('using_CV_1.pdf')