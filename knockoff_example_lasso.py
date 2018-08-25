import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(knockoff)')

from core import (infer_full_target,
                  split_sampler,
                  normal_sampler,
                  logit_fit,
                  probit_fit)

from knockoffs import knockoffs_sigma

def simulate(n=1000, p=50, signal=3.2, sigma=2, alpha=0.1, s=10):

    # description of statistical problem

    X, y, truth, _, _, sigmaX = gaussian_instance(n=n,
                                                  p=p, 
                                                  s=s,
                                                  equicorrelated=False,
                                                  rho=0., 
                                                  sigma=sigma,
                                                  signal=signal,
                                                  random_signs=True,
                                                  scale=False)

    dispersion = sigma**2
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)

    sampler = normal_sampler(X.T.dot(y), covS)
    splitting_sampler = split_sampler(X * y[:, None], covS / n)

    def meta_algorithm(XTXi, X, resid, sampler):

        min_success = 1
        ntries = 1
        p = XTXi.shape[0]
        success = np.zeros(p)

        for _ in range(ntries):
            S = sampler(scale=0) # deterministic with scale=0
            ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
            K = knockoffs_sigma(X, ynew, *[None]*4)
            K.setup(sigmaX)
            select = K.select()[0]
            print(select, 'blah')
            numpy2ri.deactivate()
            success[select] += 1
        return set(np.nonzero(success >= min_success)[0])

    selection_algorithm = functools.partial(meta_algorithm, XTXi, X, y - X.dot(XTXi.dot(S)))

    # run selection algorithm

    observed_set = selection_algorithm(splitting_sampler)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths, naive_lengths = [], [], [], []
    for idx in list(observed_set)[:1]:
        print(idx, len(observed_set))
        true_target = truth[idx]

        (pivot, 
         interval) = infer_full_target(selection_algorithm,
                                       observed_set,
                                       idx,
                                       sampler,
                                       dispersion,
                                       hypothesis=true_target,
                                       fit_probability=probit_fit,
                                       alpha=alpha,
                                       B=100)

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        naive_lengths.append(2 * ndist.ppf(1 - 0.5 * alpha) * target_sd)

    return pivots, covered, lengths, naive_lengths

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    U = np.linspace(0, 1, 101)
    P, L, N, coverage = [], [], [], []
    plt.clf()
    for i in range(1000):
        p, cover, l, n = simulate()
        coverage.extend(cover)
        P.extend(p)
        L.extend(l)
        N.extend(n)
        print(np.mean(P), np.std(P), np.mean(np.array(L) / np.array(N)), np.mean(coverage))

        if i % 5 == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P)(U), 'r', linewidth=3)
            plt.plot([0,1], [0,1], 'k--', linewidth=2)
            plt.savefig('knockoff_example_lasso.pdf')
