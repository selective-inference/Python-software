import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.core import (infer_full_target,
                                  split_sampler,
                                  normal_sampler,
                                  logit_fit,
                                  gbm_fit,
                                  repeat_selection,
                                  probit_fit)
from selection.learning.utils import pivot_plot

from selection.learning.learners import mixture_learner
mixture_learner.scales = [1]*10 + [1.5,2,3,4,5,10]

def BHfilter(pval, q=0.2):
    pval = np.asarray(pval)
    pval_sort = np.sort(pval)
    comparison = q * np.arange(1, pval.shape[0] + 1.) / pval.shape[0]
    passing = pval_sort < comparison
    if passing.sum():
        thresh = comparison[np.nonzero(passing)[0].max()]
        return np.nonzero(pval <= thresh)[0]
    return []

def simulate(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=1000):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False)[:3]

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTX, XTXi, dispersion, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        scale = 0.
        noisy_S = sampler(scale=scale)
        soln = XTXi.dot(noisy_S)
        solnZ = soln / (np.sqrt(np.diag(XTXi)) * np.sqrt(dispersion))
        pval = ndist.cdf(solnZ)
        pval = 2 * np.minimum(pval, 1 - pval)
        return set(BHfilter(pval, q=0.2))

    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, dispersion)

    # run selection algorithm

    success_params = (1, 1)

    observed_set = repeat_selection(selection_algorithm, smooth_sampler, *success_params)

    # find the target, based on the observed outcome

    # we just take the first target  

    targets = []
    idx = sorted(observed_set)
    np.random.shuffle(idx)
    idx = idx[:1]
    if len(idx) > 0:
        print("variable: ", idx, "total selected: ", len(observed_set))
        true_target = truth[idx]

        results = infer_full_target(selection_algorithm,
                                    observed_set,
                                    idx,
                                    splitting_sampler,
                                    dispersion,
                                    hypothesis=true_target,
                                    fit_probability=logit_fit,
                                    fit_args={'df':20},
                                    success_params=success_params,
                                    alpha=alpha,
                                    B=B,
                                    single=True)

        pvalues = [r[2] for r in results]
        covered = [(r[1][0] < t) * (r[1][1] > t) for r, t in zip(results, true_target)]
        pivots = [r[0] for r in results]

        target_sd = np.sqrt(np.diag(dispersion * XTXi)[idx])
        observed_target = XTXi[idx].dot(X.T.dot(y))
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = np.vstack([observed_target - quantile * target_sd, observed_target + quantile * target_sd])

        naive_pivots = (1 - ndist.cdf((observed_target - true_target) / target_sd))
        naive_pivots = 2 * np.minimum(naive_pivots, 1 - naive_pivots)

        naive_pvalues = (1 - ndist.cdf(observed_target / target_sd))
        naive_pvalues = 2 * np.minimum(naive_pvalues, 1 - naive_pvalues)

        naive_covered = (naive_interval[0] < true_target) * (naive_interval[1] > true_target)
        naive_lengths = naive_interval[1] - naive_interval[0]
        lower = [r[1][0] for r in results]
        upper = [r[1][1] for r in results]
        lengths = np.array(upper) - np.array(lower)

        return pd.DataFrame({'pivot':pivots,
                             'pvalue':pvalues,
                             'coverage':covered,
                             'length':lengths,
                             'naive_pivot':naive_pivots,
                             'naive_coverage':naive_covered,
                             'naive_length':naive_lengths,
                             'upper':upper,
                             'lower':lower,
                             'targets':true_target,
                             'batch_size':B * np.ones(len(idx), np.int)})


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(2000):
        df = simulate(B=5000)
        csvfile = 'logit_targets_BH_single.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)
