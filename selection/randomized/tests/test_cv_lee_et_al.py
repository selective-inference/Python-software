import functools

import numpy as np
from scipy.stats import norm as ndist
import matplotlib.pyplot as plt

import regreg.api as rr

import selection.api as sel
from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import lasso
from selection.randomized.cv import CV


def test_cv_lee_et_al(n=3000,
                      p=1000,
                      s=10,
                      snr = 3.5,
                      rho =0.,
                      K=5):

    print (n, p, s, rho)

    X, y, beta, truth, sigma = gaussian_instance(n=n, p=p, s=s, snr=3.5, sigma=1., scale=True, center=True)

    truth = np.nonzero(beta != 0)[0]

    lam_seq = np.exp(np.linspace(np.log(1.e-6), np.log(2), 30)) * np.fabs(X.T.dot(y)).max()
    folds = np.arange(n) % K
    np.random.shuffle(folds)

    CV_compute = CV(rr.glm.gaussian(X,y), folds, lam_seq)
    lam_CV, _, _,_ = CV_compute.choose_lambda_CVr()

    L = lasso.gaussian(X, y, lam_CV)
    soln = L.fit()

    active = soln != 0


    if set(truth).issubset(np.nonzero(active)[0]) and active.sum() > 0:

        print("nactive", active.sum())
        active_set = np.nonzero(active)[0]

        # Lee et al. using sigma
        P0_lee, PA_lee = [], []
        L0 = lasso.gaussian(X, y, lam_CV, sigma=sigma)
        L0.fit()

        for i in range(active.sum()):

            keep = np.zeros(active.sum())
            keep[i] = 1.
            pivot = L0.constraints.pivot(keep,
                                        L.onestep_estimator,
                                        alternative='twosided')

            if active_set[i] in truth:
                PA_lee.append(pivot)
            else:
                P0_lee.append(pivot)

        return  P0_lee, PA_lee
    else:
        return [], []


if __name__ == "__main__":
    np.random.seed(500)
    P0_lee, PA_lee = [], []

    for i in range(100):
        print("iteration", i)
        p0_lee, pA_lee = test_cv_lee_et_al(n=50, p=20, s=0, snr=3.5, K=5, rho=0.)
        P0_lee.extend(p0_lee); PA_lee.extend(pA_lee)
        print (np.mean(P0_lee), np.std(P0_lee), np.mean(np.array(P0_lee)<0.05), 'null lee')
        print (np.mean(PA_lee), np.std(PA_lee), 'alt lee')

        if len(P0_lee) > 0:
            import statsmodels.api as sm
            import matplotlib.pyplot as plt
            U = np.linspace(0,1,101)
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P0_lee)(U), label='Lee et al.')
            plt.xlabel("observed p-value")
            plt.ylabel("CDF")
            plt.suptitle("P-values using Lee et al. truncated Gaussian")
            plt.plot([0,1],[0,1], 'k--')
            plt.legend(loc='lower right')
            plt.savefig('using_CV.pdf')