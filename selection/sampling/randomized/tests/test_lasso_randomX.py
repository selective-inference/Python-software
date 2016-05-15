import numpy as np
from scipy.stats import laplace, probplot, uniform

import selection.sampling.randomized.losses.lasso_randomX as lasso_randomX
reload(lasso_randomX)

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
reload(randomized)
from pvalues_new import pval_new
from matplotlib import pyplot as plt


def test_lasso_randomX(s=5, n=100, p=30):

    X, y, true_beta, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1., rho=0)

    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = lasso_randomX.lasso_randomX(X, y)
    random_Z = randomization.rvs(p)
    epsilon = 1.

    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    # this is the initial randomization based on which the active set is chosen
    # done is sampler.py

    random_Z = randomization.rvs(p)

    penalty = randomized.selective_l1norm_new(p, lagrange=lam)

    sampler1 = randomized.selective_sampler_MH_new(loss,
                                               random_Z,
                                               epsilon,
                                               randomization,
                                               penalty)

    sampler1.loss.fit_E(sampler1.penalty.active_set) # also calls bootstrap_covariance function in this class

    # new

    active_set = sampler1.penalty.active_set # set E
    print 'size of the active set', np.sum(active_set)
    beta_unpenalized = sampler1.loss._beta_unpenalized  #\bar{\beta}_E

    # the estimate of the covariance of \bar{\beta}_E-\beta_E = (X_E^TX_E)^{-1}X_E^T\epsilon
    # is Sigma=(X_E^TX_E)^{-1}COV(X_E^T\epsilon), where COV(X_E^T\epsilon) is the pairs bootstrap estimate
    #Sigma = np.dot(np.linalg.inv(sampler1.loss._XETXE), sampler1.loss._cov_XETepsilon)

    # bootstrapped covariance of \bar{\beta}_E, (X_E^{*T}X_E^*)^{-1}X_E^{*T}(y^*-X_E^*\bar{\beta}_E)
    Sigma_b = sampler1.loss._cov_beta_bar
    #print Sigma_b.diagonal()

    #Sigma_oracle = np.linalg.inv(sampler1.loss._XETXE)


    inactive_set = ~active_set

    residual = y - np.dot(X[:,active_set], beta_unpenalized)  # y-X_E\bar{\beta}^E

    N = np.dot(X[:,inactive_set].T,residual)  # X_{-E}^T(y-X_E\bar{\beta}_E), null statistic

    linear_part = np.identity(p) # not used anymore

    data = np.concatenate((beta_unpenalized, N))  # (\bar{\beta}_E, X_{-E}^T(y-X_E\bar{\beta}^E))

    loss_args = {'beta':true_beta.copy()}

    # linear_part=np.identity(p)
    # data=np.dot(X.T, y)

    # pval_new_new function calls sampler.setup_sampling and runs the sampling scheme
    null, alt = pval_new(sampler1,
                         loss_args,
                         linear_part,
                         data,
                         nonzero,
                         Sigma_b,
                         true_beta)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(5):
        print "iteration", i
        p0, pA = test_lasso_randomX()
        P0.extend(p0); PA.extend(pA)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.show()
