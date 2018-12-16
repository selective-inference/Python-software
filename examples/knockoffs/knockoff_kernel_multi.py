import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import ROSI

from learn_selection.core import (infer_full_target,
                                  split_sampler, 
                                  normal_sampler,
                                  logit_fit,
                                  repeat_selection,
                                  probit_fit)
from learn_selection.keras_fit import keras_fit

def simulate(n=1000, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, seed=0):

    # description of statistical problem

    np.random.seed(seed)
    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False,
                                    center=False)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)

    def meta_algorithm(X, XTXi, resid, sampler):

        n, p = X.shape

        rho = 0.8
        S = sampler(scale=0.) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        Xnew = rho * X + np.sqrt(1 - rho**2) * np.random.standard_normal(X.shape)

        X_full = np.hstack([X, Xnew])
        beta_full = np.linalg.pinv(X_full).dot(ynew)
        winners = np.fabs(beta_full)[:p] > np.fabs(beta_full)[p:]
        return set(np.nonzero(winners)[0])

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

    # run selection algorithm

    success_params = (8, 10)

    observed_set = repeat_selection(selection_algorithm, smooth_sampler, *success_params)

    # find the target, based on the observed outcome

    # find the target, based on the observed outcome

    pivots, covered, lengths, pvalues = [], [], [], []
    lower, upper = [], []
    naive_pvalues, naive_pivots, naive_covered, naive_lengths =  [], [], [], []

    targets = []
    true_target = truth[sorted(observed_set)]

    if len(observed_set) > 0:
        results = infer_full_target(selection_algorithm,
                                    observed_set,
                                    sorted(observed_set),
                                    smooth_sampler,
                                    dispersion,
                                    hypothesis=true_target,
                                    fit_probability=keras_fit,
                                    fit_args={'epochs':20, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'},
                                    success_params=success_params,
                                    alpha=alpha,
                                    B=3000)

        for i, result in enumerate(results):

            (pivot, 
             interval,
             pvalue,
             _) = result

            pvalues.append(pvalue)
            pivots.append(pivot)
            covered.append((interval[0] < true_target[i]) * (interval[1] > true_target[i]))
            lengths.append(interval[1] - interval[0])

        for i, idx in enumerate(sorted(observed_set)):
            target_sd = np.sqrt(dispersion * XTXi[idx, idx])
            observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
            quantile = ndist.ppf(1 - 0.5 * alpha)
            naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)

            naive_pivot = (1 - ndist.cdf((observed_target - true_target[i]) / target_sd))
            naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
            naive_pivots.append(naive_pivot)

            naive_pvalue = (1 - ndist.cdf(observed_target / target_sd))
            naive_pvalue = 2 * min(naive_pvalue, 1 - naive_pvalue)
            naive_pvalues.append(naive_pvalue)

            naive_covered.append((naive_interval[0] < true_target[i]) * (naive_interval[1] > true_target[i]))
            naive_lengths.append(naive_interval[1] - naive_interval[0])
            lower.append(interval[0])
            upper.append(interval[1])

    if len(pvalues) > 0:
        return pd.DataFrame({'pivot':pivots,
                             'pvalue':pvalues,
                             'coverage':(np.array(lower) < true_target) * (np.array(upper) > true_target),
                             'length':lengths,
                             'naive_pivot':naive_pivots,
                             'naive_coverage':naive_covered,
                             'naive_length':naive_lengths,
                             'upper':upper,
                             'lower':lower,
                             'target':true_target
                             })


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    iseed = int(np.fabs(np.random.standard_normal() * 50000))
    for i in range(500):
        df = simulate(seed=i + iseed)
        csvfile = 'knockoff_kernel_multi.csv'

        if df is not None and i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass

            df['coverage'] = (df['lower'] < df['target']) * (df['upper'] > df['target'])  
            if len(df['pivot']) > 0:

                print("selective:", np.mean(df['pivot']), np.std(df['pivot']), np.mean(df['length']), np.std(df['length']), np.mean(df['coverage']))
                print("naive:", np.mean(df['naive_pivot']), np.std(df['naive_pivot']), np.mean(df['naive_length']), np.std(df['naive_length']), np.mean(df['naive_coverage']))

                print("len ratio selective divided by naive:", np.mean(np.array(df['length']) / np.array(df['naive_length'])))

                plt.clf()
                U = np.linspace(0, 1, 101)
                plt.plot(U, sm.distributions.ECDF(df['pivot'])(U), 'r', label='Selective', linewidth=3)
                plt.plot(U, sm.distributions.ECDF(df['naive_pivot'])(U), 'b', label='Naive', linewidth=3)
                plt.legend()
                plt.plot([0,1], [0,1], 'k--', linewidth=2)
                plt.savefig(csvfile[:-4] + '.pdf')

                plt.clf()
                plt.scatter(df['naive_length'], df['length'])
                plt.savefig(csvfile[:-4] + '_lengths.pdf')

            df.to_csv(csvfile, index=False)

