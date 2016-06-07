import numpy as np
from scipy.stats import laplace, probplot, uniform, norm

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues_high_dim import pval_high_dim
from matplotlib import pyplot as plt

def test_lasso(s=0, n=200, p=100):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    print 'sigma', sigma
    lam_frac = 1.

    randomization = norm(loc=0, scale=1./np.sqrt(n))
    loss = randomized.gaussian_Xfixed_high_dim(X, y)
    random_Z = randomization.rvs(p)
    epsilon = 1.

    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    random_Z = randomization.rvs(p)
    penalty = randomized.selective_l1norm_high_dim(p, lagrange=lam)

    sampler1 = randomized.selective_sampler_MH_high_dim(loss,
                                               random_Z,
                                               epsilon,
                                               randomization,
                                               penalty)

    loss_args = {'mean':np.zeros(n),
                 'sigma':sigma}

    data = np.dot(X.T, y)

    null, alt = pval_high_dim(sampler1,
                     loss_args,
                     X, data,
                     nonzero)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(500):
        print "iteration", i
        p0, pA = test_lasso()
        P0.extend(p0); PA.extend(pA)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.show()
