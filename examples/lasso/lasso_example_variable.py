import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import lasso, ROSI
from knockoffs import lasso_glmnet

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(knockoff); library(glmnet)')
from rpy2 import rinterface

from core import (infer_full_target,
                  split_sampler, # split_sampler not working yet
                  normal_sampler,
                  logit_fit,
                  probit_fit)

def simulate(n=200, p=100, s=10, signal=(0, 0), sigma=2, alpha=0.1):

    # description of statistical problem

    while True:

        X, y, truth = gaussian_instance(n=n,
                                        p=p, 
                                        s=s,
                                        equicorrelated=False,
                                        rho=0.5, 
                                        sigma=sigma,
                                        signal=signal,
                                        random_signs=True,
                                        scale=False)[:3]

        dispersion = sigma**2

        S = X.T.dot(y)
        covS = dispersion * X.T.dot(X)
        smooth_sampler = normal_sampler(S, covS)
        splitting_sampler = split_sampler(X * y[:, None], covS)

        def meta_algorithm(X, XTXi, resid, sampler):

            S = sampler(scale=0.) # deterministic with scale=0
            ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
            G = lasso_glmnet(X, ynew, *[None]*4)
            select = G.select(CV=False)
            return set(list(select[0]))

        XTX = X.T.dot(X)
        XTXi = np.linalg.inv(XTX)
        resid = y - X.dot(XTXi.dot(X.T.dot(y)))
        dispersion = np.linalg.norm(resid)**2 / (n-p)

        selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

        # run selection algorithm

        observed_set = selection_algorithm(smooth_sampler)

        # find the target, based on the observed outcome

        # we just take the first target  

        pivots, covered, lengths = [], [], []
        naive_pivots, naive_covered, naive_lengths =  [], [], []

        for idx in list(observed_set)[:1]:
            print("variable: ", idx, "total selected: ", len(observed_set))
            true_target = truth[idx]

            (pivot, 
             interval) = infer_full_target(selection_algorithm,
                                           observed_set,
                                           idx,
                                           splitting_sampler,
                                           dispersion,
                                           hypothesis=true_target,
                                           fit_probability=probit_fit,
                                           alpha=alpha,
                                           B=2000)

            pivots.append(pivot)
            covered.append((interval[0] < true_target) * (interval[1] > true_target))
            lengths.append(interval[1] - interval[0])

            target_sd = np.sqrt(dispersion * XTXi[idx, idx])
            observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
            quantile = ndist.ppf(1 - 0.5 * alpha)
            naive_interval = (observed_target-quantile * target_sd, observed_target+quantile * target_sd)
            naive_pivot = (1-ndist.cdf((observed_target-true_target)/target_sd))
            naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
            naive_pivots.append(naive_pivot) # one-sided

            naive_covered.append((naive_interval[0]<true_target)*(naive_interval[1]>true_target))
            naive_lengths.append(naive_interval[1]-naive_interval[0])

        if len(observed_set) > 0: # we've found one
            break

    while True:

        X, y, truth = gaussian_instance(n=n,
                                        p=p, 
                                        s=s,
                                        equicorrelated=False,
                                        rho=0.5, 
                                        sigma=sigma,
                                        signal=signal,
                                        random_signs=True,
                                        scale=False)[:3]


        numpy2ri.activate()
        rpy.r.assign('X', X)
        rpy.r.assign('Y', y)
        rpy.r('X = as.matrix(X)')
        rpy.r('Y = as.numeric(Y)')
        rpy.r('cvG = cv.glmnet(X, Y, intercept=FALSE, standardize=FALSE)')
        lam = rpy.r("0.99 * cvG[['lambda.1se']]")[0]
        numpy2ri.deactivate()

        print(n * lam, np.fabs(X.T.dot(y)).max(), 'duh')
        R = ROSI.gaussian(X, y, n * lam, sigma=np.sqrt(dispersion), approximate_inverse=None)
        R.fit()
        S = R.summary()
        if S is not None:
            print('blah')
            pv = np.array(S['pval'])[0]
            pv = 2 * min(pv, 1 - pv)
            liu_pvalues = [pv]
            break

    return pivots, covered, lengths, naive_pivots, naive_covered, naive_lengths, liu_pvalues


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    P, L, coverage = [], [], []
    naive_P, naive_L, naive_coverage = [], [], []
    plt.clf()
    for i in range(500):
        p, cover, l, naive_p, naive_covered, naive_l, liu = simulate()

        csvfile = 'lasso_1se.csv'

        if i > 0:

            df = pd.DataFrame({'pivot':p,
                               'naive_pivot':naive_p,
                               'coverage':cover,
                               'naive_coverage':naive_covered,
                               'length':l,
                               'naive_length':naive_l,
                               'liu_pivots':liu})
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
                if (~np.isnan(df['liu_pivots'])).sum() > 0:
                    plt.plot(U, sm.distributions.ECDF(df['liu_pivots'][~np.isnan(df['liu_pivots'])])(U), 'g', label='Liu', linewidth=3)
                plt.legend()
                plt.plot([0,1], [0,1], 'k--', linewidth=2)
                plt.savefig('lasso_example_1se.pdf')

            df.to_csv(csvfile, index=False)

