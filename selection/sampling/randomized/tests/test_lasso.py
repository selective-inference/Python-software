import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues import pval
from matplotlib import pyplot as plt

def test_lasso(s=5, n=100, p=20):
    
    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    print 'sigma', sigma
    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.gaussian_Xfixed(X, y)
    random_Z = randomization.rvs(p)
    epsilon = 1.

    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm_new(p, lagrange=lam)

    sampler1 = randomized.selective_sampler_MH_new(loss,
                                               random_Z,
                                               epsilon,
                                               randomization,
                                               penalty)

    loss_args = {'mean':np.zeros(n), 
                 'sigma':sigma}

    null, alt = pval(sampler1,
                     loss_args,
                     X, y, y,
                     nonzero)
    
    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(30):
        print "iteration", i
        p0, pA = test_lasso()
        P0.extend(p0); PA.extend(pA)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.show()
