import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import ROSI
from learn_selection.knockoffs import cv_glmnet_lam, lasso_glmnet

from learn_selection.core import (infer_full_target,
                                  split_sampler,
                                  normal_sampler,
                                  logit_fit,
                                  repeat_selection,
                                  probit_fit)

def simulate(n=400, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, seed=0):

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

    lam_min, lam_1se = cv_glmnet_lam(X, y)
    lam_min, lam_1se = n * lam_min, n * lam_1se

    def meta_algorithm(X, XTXi, resid, sampler):

        n, p = X.shape
        idx = np.random.choice(np.arange(n), 200, replace=False)

        S = sampler(scale=0.) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X

        G = lasso_glmnet(X[idx], ynew[idx], *[None]*4)
        select = G.select()
        return set(list(select[0]))

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

    # run selection algorithm

    success_params = (1, 1)

    observed_set = repeat_selection(selection_algorithm, smooth_sampler, *success_params)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths, pvalues = [], [], [], []
    lower, upper = [], []
    naive_pvalues, naive_pivots, naive_covered, naive_lengths =  [], [], [], []

    targets = []

    for idx in sorted(observed_set)[:1]:
        print("variable: ", idx, "total selected: ", len(observed_set))
        true_target = [truth[idx]]
        targets.extend(true_target)

        np.random.seed(seed)
        X2, _, _ = gaussian_instance(n=n,
                                     p=p, 
                                     s=s,
                                     equicorrelated=False,
                                     rho=0.5, 
                                     sigma=sigma,
                                     signal=signal,
                                     random_signs=True,
                                     center=False,
                                     scale=False)[:3]
        stage_1 = np.random.choice(np.arange(n), 200, replace=False)

        stage_2 = sorted(set(range(n)).difference(stage_1))
        X2 = X2[stage_2]
        y2 = X2.dot(truth) + sigma * np.random.standard_normal(X2.shape[0])

        XTXi_2 = np.linalg.inv(X2.T.dot(X2))
        resid2 = y2 - X2.dot(XTXi_2.dot(X2.T.dot(y2)))
        dispersion_2 = np.linalg.norm(resid2)**2 / (X2.shape[0] - X2.shape[1])

        target_sd = np.sqrt(dispersion_2 * XTXi_2[idx, idx])
        observed_target = np.squeeze(XTXi_2[idx].dot(X2.T.dot(y2)))
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)

        naive_pivot = (1 - ndist.cdf((observed_target - true_target[0]) / target_sd))
        naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pivots.append(naive_pivot)

        naive_pvalue = (1 - ndist.cdf(observed_target / target_sd))
        naive_pvalue = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pvalues.append(naive_pvalue)

        naive_covered.append((naive_interval[0] < true_target[0]) * (naive_interval[1] > true_target[0]))
        naive_lengths.append(naive_interval[1] - naive_interval[0])

        (pivot, 
         interval,
         pvalue,
         _) = infer_full_target(selection_algorithm,
                                observed_set,
                                [idx],
                                smooth_sampler,
                                dispersion,
                                hypothesis=true_target,
                                fit_probability=probit_fit,
                                success_params=success_params,
                                alpha=alpha,
                                B=1000)[0]

        pvalues.append(pvalue)
        pivots.append(pivot)
        covered.append((interval[0] < true_target[0]) * (interval[1] > true_target[0]))
        print(interval, 'interval')
        lengths.append(interval[1] - interval[0])
        lower.append(interval[0])
        upper.append(interval[1])

    if len(pvalues) > 0:
        return pd.DataFrame({'pivot':pivots,
                             'target':targets,
                             'pvalue':pvalues,
                             'coverage':covered,
                             'length':lengths,
                             'naive_pivot':naive_pivots,
                             'naive_coverage':naive_covered,
                             'naive_length':naive_lengths,
                             'upper':upper,
                             'lower':lower})


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    iseed = int(np.fabs(np.random.standard_normal() * 1000))
    for i in range(500):
        df = simulate(seed=i + iseed)
        csvfile = 'posthoc.csv'

        if df is not None and i % 2 == 1 and i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass

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

