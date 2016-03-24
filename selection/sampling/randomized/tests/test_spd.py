import numpy as np
from scipy.stats import laplace, norm, probplot
from sklearn.datasets import make_sparse_spd_matrix

from selection.algorithms.randomized import logistic_instance
import selection.sampling.randomized.api as randomized
from pvalues import pval

def main(rho=0.245, n=100, p=30):
    
    X, prec, nonzero = instance(n=n, p=p, alpha=0.99, rho=rho)
    lam_frac = 0.1
    alpha = 0.8

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.neighbourhood_selection(X) 
    epsilon = 1.

    lam = 2./np.sqrt(n) * np.linalg.norm(X) * norm.isf(alpha / (2 * p**2))

    random_Z = randomization.rvs(p**2 - p)
    penalty = randomized.selective_l1norm(p**2-p, lagrange=lam)

    sampler1 = randomized.selective_sampler_MH(loss,
                                               random_Z,
                                               epsilon,
                                               randomization,
                                               penalty)

    loss_args = {"active":sampler1.penalty.active_set,
                 "quadratic_coef":epsilon}
    null, alt = pval(sampler1, 
                     loss_args,
                     None, X,
                     nonzero)
    
    return null, alt

def instance(n, p, alpha, rho):
    # Generate the data
    prec = make_sparse_spd_matrix(p, alpha=alpha,
                                  smallest_coef=rho,
                                  largest_coef=rho,
                                  norm_diag=True)
    off_diagonal = ~np.identity(p, dtype=bool)
    nonzero = np.where(prec[off_diagonal] != 0)[0] 

    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    X /= np.sqrt(n)

    return X, prec, nonzero 

if __name__ == "__main__":
    
    P0, PA = [], []
    for i in range(100):
        print "iteration", i
        p0, pA = main()
        P0.extend(p0); PA.extend(pA)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    probplot(P0, dist=stats.uniform, sparams=(0,1), plot=plt, fit=True)
    plt.show()
