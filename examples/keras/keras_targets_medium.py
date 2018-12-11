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
                                  gbm_fit,
                                  repeat_selection,
                                  probit_fit)

from learn_selection.keras_fit import keras_fit
from learn_selection.learners import mixture_learner
mixture_learner.scales = [1]*10 + [1.5,2,3,4,5,10]

def simulate(n=200, p=50, s=5, signal=(0.5, 1), sigma=2, alpha=0.1, B=1000):

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

    def meta_algorithm(XTX, XTXi, dispersion, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.5
        noisy_S = sampler(scale=scale)
        soln = XTXi.dot(noisy_S)
        solnZ = soln / (np.sqrt(np.diag(XTXi)) * np.sqrt(dispersion))
        return set(np.nonzero(np.fabs(solnZ) > 2.1)[0])

    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, dispersion, lam)

    # run selection algorithm

    success_params = (1, 1)

    observed_set = repeat_selection(selection_algorithm, smooth_sampler, *success_params)

    # find the target, based on the observed outcome

    # we just take the first target  

    targets = []
    idx = sorted(observed_set)
    if len(idx) > 0:
        print("variable: ", idx, "total selected: ", len(observed_set))
        true_target = truth[idx]

        results = infer_full_target(selection_algorithm,
                                    observed_set,
                                    idx,
                                    splitting_sampler,
                                    dispersion,
                                    hypothesis=true_target,
                                    fit_probability=keras_fit,
                                    fit_args={'epochs':30, 'sizes':[100, 100], 'activation':'relu'},
                                    success_params=success_params,
                                    alpha=alpha,
                                    B=B)

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

    U = np.linspace(0, 1, 101)
    plt.clf()

    for i in range(500):
        df = simulate(B=10000)
        csvfile = 'keras_targets_medium.csv'

        try:
            df = pd.concat([df, pd.read_csv(csvfile)])
        except FileNotFoundError:
            pass

        if df is not None and len(df['pivot']) > 0:

            print(df['pivot'], 'pivot')
            plt.clf()
            U = np.linspace(0, 1, 101)
            plt.plot(U, sm.distributions.ECDF(df['naive_pivot'])(U), 'b', label='Naive', linewidth=3)
            for b in np.unique(df['batch_size']):
                plt.plot(U, sm.distributions.ECDF(np.array(df['pivot'])[np.array(df['batch_size']) == b])(U), label='B=%d' % b, linewidth=3)

            plt.legend()
            plt.plot([0,1], [0,1], 'k--', linewidth=2)
            plt.savefig(csvfile[:-4] + '.pdf')

            df.to_csv(csvfile, index=False)

