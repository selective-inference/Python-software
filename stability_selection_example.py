import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from core import (infer_full_target,
                  split_sampler, # split_sampler not working yet
                  normal_sampler,
                  logit_fit,
                  probit_fit)

from sklearn.linear_model import lasso_path

def simulate(n=200, p=100, s=20, signal=(2, 2), sigma=2, alpha=0.1):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p,
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.1,
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=True)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTX, XTXi, sampler):

        min_success = 3
        ntries = 5

        def _alpha_grid(X, y, center, XTX):
            n, p = X.shape
            alphas, coefs, _ = lasso_path(X, y, Xy=center, precompute=XTX)
            nselected = np.count_nonzero(coefs, axis=0)
            return alphas[nselected < np.sqrt(0.8 * p)]

        alpha_grid = _alpha_grid(X, y, sampler(scale=0.), XTX)

        success = np.zeros((p, alpha_grid.shape[0]))

        for _ in range(ntries):
            scale = 1.  # corresponds to sub-samples of 50%
            noisy_S = sampler(scale=scale)
            _, coefs, _ = lasso_path(X, y, Xy = noisy_S, precompute=XTX, alphas=alpha_grid)
            success += np.abs(np.sign(coefs))

        selected = np.apply_along_axis(lambda row: any(x>min_success for x in row), 1, success)
        vars = set(np.nonzero(selected)[0])
        return vars

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)

    #alpha_grid = _alpha_grid(X, y, XTX) # decreasing
    #print("alpha grid length:", alpha_grid.shape)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi)

    # run selection algorithm

    observed_set = selection_algorithm(smooth_sampler)

    print("observed set",observed_set)
    print("observed and true", observed_set.intersection(set(np.nonzero(truth!=0)[0])))
    print("observed and false", observed_set.intersection(set(np.nonzero(truth==0)[0])))
    # find the target, based on the observed outcome

    # we just take the first target

    pivots, covered, lengths = [], [], []
    naive_pivots, naive_covered, naive_lengths =  [], [], []

    for idx in observed_set:
        print("variable: ", idx, "total selected: ", len(observed_set))
        true_target = truth[idx]

        (pivot,
         interval) = infer_full_target(selection_algorithm,
                                       observed_set,
                                       idx,
                                       splitting_sampler,
                                       dispersion,
                                       hypothesis=true_target,
                                       fit_probability=logit_fit,
                                       alpha=alpha,
                                       B=1000)

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
        quantile = ndist.ppf(1 - alpha)
        naive_interval = (observed_target-quantile * target_sd, observed_target+quantile * target_sd)
        naive_pivot = (1-ndist.cdf((observed_target-true_target)/target_sd)) # one-sided
        naive_pivot = 2*min(1-naive_pivot, naive_pivot)
        naive_pivots.append(naive_pivot) # two-sided

        naive_covered.append((naive_interval[0]<true_target)*(naive_interval[1]>true_target))
        naive_lengths.append(naive_interval[1]-naive_interval[0])

    return pivots, covered, lengths, naive_pivots, naive_covered, naive_lengths


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pickle

    fit_label = "logit"
    seedn = 2
    outfile = "".join([fit_label, str(seedn), ".pkl"])
    np.random.seed(seedn)

    U = np.linspace(0, 1, 101)
    P, L, coverage = [], [], []
    naive_P, naive_L, naive_coverage = [], [], []
    plt.clf()

    for i in range(30):
        p, cover, l, naive_p, naive_covered, naive_l = simulate()
        coverage.extend(cover)
        P.extend(p)
        L.extend(l)
        naive_P.extend(naive_p)
        naive_coverage.extend(naive_covered)
        naive_L.extend(naive_l)

        print("selective:", np.mean(P), np.std(P), np.mean(L) , np.mean(coverage))
        print("naive:", np.mean(naive_P), np.std(naive_P), np.mean(naive_L), np.mean(naive_coverage))
        print("len ratio selective divided by naive:", np.mean(np.array(L) / np.array(naive_L)))

        if i % 5 == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P)(U), 'r', linewidth=3, label='Selective')
            plt.plot(U, sm.distributions.ECDF(naive_P)(U), 'b', linewidth=3, label='Naive')
            plt.plot([0,1], [0,1], 'k--', linewidth=2)
            plt.legend()
            plt.savefig('lasso_example4.pdf')

    with open(outfile, "wb") as f:
        pickle.dump((coverage, P, L, naive_coverage, naive_P, naive_L), f)
