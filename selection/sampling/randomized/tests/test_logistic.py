import numpy as np
from scipy.stats import laplace, probplot, uniform
import matplotlib.pyplot as plt

from selection.algorithms.randomized import logistic_instance
import selection.sampling.randomized.api as randomized
from pvalues import pval

@np.testing.decorators.setastest(False)
def test_logistic(s=5, n=200, p=20):
    
    X, y, beta, active= logistic_instance(n=n, p=p, s=s, rho=0)
    nonzero = np.where(beta)[0]
    lam_frac = 40.8

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.logistic_Xrandom(X, y)
    epsilon = 1.

    #lam = lam_frac * np.mean(np.fabs(np.dot(X.T, (np.random.binomial(1, 1./2, (n, 10000)) - 0.5))).max(0))
    lam = 70.
    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm(p, lagrange=lam)

    sampler1 = randomized.selective_sampler_MH(loss,
                                               random_Z,
                                               epsilon,
                                               randomization,
                                               penalty)

    sampler1.loss.fit_E(sampler1.penalty.active_set)
    linear_part = np.identity(p)
    data = np.dot(X.T, y - 1./2)

    loss_args = {'mean':np.zeros(p)}

    null, alt = pval(sampler1,
                     loss_args,
                     linear_part,
                     data, 
                     nonzero)
    
    return null, alt

if __name__ == "__main__":
    
    P0, PA = [], []
    for i in range(10):
        print "iteration", i
        p0, pA = test_logistic()
        P0.extend(p0); PA.extend(pA)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=stats.uniform, sparams=(0,1), plot=plt, fit=True)
    plt.show()
