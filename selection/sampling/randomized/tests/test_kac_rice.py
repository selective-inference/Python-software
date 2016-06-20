import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues import pval
from matplotlib import pyplot as plt
from selection.distributions.discrete_family import discrete_family


def test_kac_rice(s=0, n=100, p=10):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    epsilon = 1.
    randomization = laplace(loc=0, scale=1.)


    T = np.dot(X.T,y)
    T_abs = np.abs(T)
    #print T
    obs = np.max(T_abs)
    j_star = np.argmax(T_abs)
    s_star = np.sign(T[j_star])
    eta_star = np.zeros(p)
    eta_star[j_star] = s_star
    #print 'maximizer', eta_star
    #print 'index, sign', j_star, s_star

    random_Z = randomization.rvs(p)

    subgrad = T + random_Z + epsilon*eta_star

    sampler = randomized.kac_rice_sampler(epsilon,
                                          randomization)

    A,b = form_Ab(j_star, p)
    sampler.setup_sampling(X, y, eta_star, subgrad, A,b)

    samples = sampler.sampling()


    pop = [np.max(np.abs(np.dot(X.T,z))) for z, _,_, in samples]
    fam = discrete_family(pop, np.ones_like(pop))
    pval = fam.cdf(0, obs)
    pval = 2 * min(pval, 1 - pval)

    print 'pvalue:', pval
    return pval

def form_Ab(j, p):
    A = np.zeros((2*p-1,p))
    A[:, j] = 1

    for i in range(j):
        A[2*i, i] = -1
        A[2*i+1, i] =1
    for i in range(j+1, p):
        A[ 2*i-1,i]= -1
        A[ 2*i,i] = 1

    b = np.zeros(2*p-1)
    return -A, b


if __name__ == "__main__":

    P0 = []
    for i in range(50):
        print "iteration", i
        #print form_Ab(1,4)
        pval = test_kac_rice()
        P0.append(pval)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.show()


