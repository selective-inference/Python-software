import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.randomized import logistic_instance
import selection.sampling.randomized.api as randomized
from matplotlib import pyplot as plt
from pvalues_logistic import pval_logistic


def test_logistic_new(s=5, n=2500, p=20):

    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0)
    #print beta
    nonzero = np.where(beta)[0]
    lam_frac = 0.8

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.logistic_Xrandom_new(X, y)
    epsilon = 1.

    lam = 400*lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1./2, (n, 10000)))).max(0))
    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm_lan_logistic(p, lagrange=lam)

    sampler1 = randomized.selective_sampler_MH_lan_logistic(loss,
                                               random_Z,
                                               epsilon,
                                               randomization,
                                               penalty)

    print 'active set size', np.sum(sampler1.penalty.active_set)
    sampler1.loss.fit_E(sampler1.penalty.active_set)

    ###### new
    active_set = sampler1.penalty.active_set
    beta_unpenalized = sampler1.loss._beta_unpenalized
    inactive_set = ~active_set

    #print 'inactive', np.sum(inactive_set)
    Sigma = sampler1.loss._cov_beta_bar

    #print 'beta unpenalized', beta_unpenalized

    w = np.exp(np.dot(X[:,active_set], beta_unpenalized))
    pi = w/(1+w)
    N = np.dot(X[:,inactive_set].T, y-pi)

    #print np.dot(N, beta_unpenalized)
    #print beta_unpenalized
    #print "N", N

    #
    linear_part = np.identity(p)
    data = np.concatenate((beta_unpenalized, N), axis=0)
    # data = np.dot(X.T, y - 1./2)

    loss_args = {'mean':np.zeros(p)}

    null, alt = pval_logistic(sampler1,
                     loss_args,
                     linear_part,
                     data,
                     nonzero,
                     Sigma)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(20):
        print "iteration", i
        p0, pA = test_logistic_new()
        P0.extend(p0); PA.extend(pA)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.show()


