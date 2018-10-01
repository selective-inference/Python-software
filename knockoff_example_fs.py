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

def simulate(n=200, p=50, signal=(2, 3), sigma=2, alpha=0.1, s=10):

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
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTXi, X, resid, sampler):

        min_success = 5
        ntries = 10
        p = XTXi.shape[0]
        success = np.zeros(p)

        for _ in range(ntries):
            S = sampler(scale=0.) # deterministic with scale=0
            ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
            K = knockoffs_sigma(X, ynew, *[None]*4)
            K.setup(sigmaX)
            K.forward_step = True
            select = K.select()[0]
            numpy2ri.deactivate()
            success[select] += 1
        value = set(np.nonzero(success >= min_success)[0])
        #print(value)
        return value

    selection_algorithm = functools.partial(meta_algorithm, XTXi, X, y - X.dot(XTXi.dot(S)))

    # run selection algorithm

    observed_set = selection_algorithm(splitting_sampler)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths = [], [], []
    naive_pivots, naive_covered, naive_lengths = [], [], []

    for idx in list(observed_set)[:1]:
        print("variable: ", idx, "total selected: ", len(observed_set))
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
                                       B=500)

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
        quantile = ndist.ppf(1 - alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)
        naive_pivot = (1 - ndist.cdf((observed_target - true_target) / target_sd))  # one-sided
        naive_pivot = 2 * min(1 - naive_pivot, naive_pivot)
        naive_pivots.append(naive_pivot)  # two-sided

        naive_covered.append((naive_interval[0] < true_target) * (naive_interval[1] > true_target))
        naive_lengths.append(naive_interval[1] - naive_interval[0])

    return pivots, covered, lengths, naive_pivots, naive_covered, naive_lengths


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pickle

    fit_label = "kk_probit"
    seedn = 2
    outfile = "".join([fit_label, str(seedn), ".pkl"])
    np.random.seed(seedn)

    U = np.linspace(0, 1, 101)
    P, L, coverage = [], [], []
    naive_P, naive_L, naive_coverage = [], [], []
    plt.clf()

    for i in range(50):
        p, cover, l, naive_p, naive_covered, naive_l = simulate()
        coverage.extend(cover)
        P.extend(p)
        L.extend(l)
        naive_P.extend(naive_p)
        naive_coverage.extend(naive_covered)
        naive_L.extend(naive_l)

        print("selective:", np.mean(P), np.std(P), np.mean(L), np.mean(coverage))
        print("naive:", np.mean(naive_P), np.std(naive_P), np.mean(naive_L), np.mean(naive_coverage))
        print("len ratio selective divided by naive:", np.mean(np.array(L) / np.array(naive_L)))

        if i % 2 == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P)(U), 'r', linewidth=3)
            plt.plot(U, sm.distributions.ECDF(naive_P)(U), 'b', linewidth=3)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.legend()
            plt.savefig('lasso_example5.pdf')

    with open(outfile, "wb") as f:
        pickle.dump((coverage, P, L, naive_coverage, naive_P, naive_L), f)
