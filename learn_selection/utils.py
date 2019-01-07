import hashlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm as normal_dbn

from selection.algorithms.lasso import ROSI

from .core import (infer_full_target,
                   split_sampler, # split_sampler not working yet
                   normal_sampler,
                   logit_fit,
                   repeat_selection,
                   probit_fit)
from .keras_fit import keras_fit

def full_model_inference(X, 
                         y,
                         truth,
                         selection_algorithm,
                         sampler,
                         success_params=(1, 1),
                         fit_probability=keras_fit,
                         fit_args={'epochs':10, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'},
                         alpha=0.1,
                         B=2000):

    # for naive inference

    n, p = X.shape
    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n - p)
                         
    instance_hash = hashlib.md5()
    instance_hash.update(X.tobytes())
    instance_hash.update(y.tobytes())
    instance_hash.update(truth.tobytes())
    instance_id = instance_hash.hexdigest()

    # run selection algorithm

    observed_set = repeat_selection(selection_algorithm, sampler, *success_params)

    # find the target, based on the observed outcome

    pivots, covered, lengths, pvalues = [], [], [], []
    lower, upper = [], []
    naive_pvalues, naive_pivots, naive_covered, naive_lengths =  [], [], [], []

    targets = []
    true_target = truth[sorted(observed_set)]

    results = infer_full_target(selection_algorithm,
                                observed_set,
                                sorted(observed_set),
                                sampler,
                                dispersion,
                                hypothesis=true_target,
                                fit_probability=fit_probability,
                                fit_args=fit_args,
                                success_params=success_params,
                                alpha=alpha,
                                B=B)

    for i, result in enumerate(results):

        (pivot, 
         interval,
         pvalue,
         _) = result

        pvalues.append(pvalue)
        pivots.append(pivot)
        covered.append((interval[0] < true_target[i]) * (interval[1] > true_target[i]))
        lengths.append(interval[1] - interval[0])
        lower.append(interval[0])
        upper.append(interval[1])

    for idx in sorted(observed_set):
        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
        quantile = normal_dbn.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)

        naive_pivot = (1 - normal_dbn.cdf((observed_target - true_target[0]) / target_sd))
        naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pivots.append(naive_pivot)

        naive_pvalue = (1 - normal_dbn.cdf(observed_target / target_sd))
        naive_pvalue = 2 * min(naive_pvalue, 1 - naive_pvalue)
        naive_pvalues.append(naive_pvalue)

        naive_covered.append((naive_interval[0] < true_target[0]) * (naive_interval[1] > true_target[0]))
        naive_lengths.append(naive_interval[1] - naive_interval[0])

    if len(pvalues) > 0:
        return pd.DataFrame({'pivot':pivots,
                             'pvalue':pvalues,
                             'coverage':covered,
                             'length':lengths,
                             'naive_pivot':naive_pivots,
                             'naive_pvalue':naive_pvalues,
                             'naive_coverage':naive_covered,
                             'naive_length':naive_lengths,
                             'upper':upper,
                             'lower':lower,
                             'id':[instance_hash]*len(pvalues),
                             'target':truth[sorted(observed_set)]})

def pivot_plot(df, 
               outbase):

    print("selective:", np.mean(df['pivot']), np.std(df['pivot']), np.mean(df['length']), np.std(df['length']), np.mean(df['coverage']))
    print("naive:", np.mean(df['naive_pivot']), np.std(df['naive_pivot']), np.mean(df['naive_length']), np.std(df['naive_length']), np.mean(df['naive_coverage']))

    print("len ratio selective divided by naive:", np.mean(np.array(df['length']) / np.array(df['naive_length'])))

    f = plt.figure(num=1)
    plt.clf()
    U = np.linspace(0, 1, 101)
    plt.plot(U, sm.distributions.ECDF(df['pivot'])(U), 'r', label='Selective', linewidth=3)
    plt.plot(U, sm.distributions.ECDF(df['naive_pivot'])(U), 'b', label='Naive', linewidth=3)
    plt.legend()
    plt.plot([0,1], [0,1], 'k--', linewidth=2)
    plt.savefig(outbase + '.pdf')
    pivot_ax = plt.gca()

    f = plt.figure(num=2)
    plt.clf()
    plt.scatter(df['naive_length'], df['length'])
    plt.savefig(outbase + '_lengths.pdf')
    length_ax = plt.gca()

    return pivot_ax, length_ax

def liu_inference(X,
                  y,
                  truth,
                  lam,
                  alpha=0.1):

    R = ROSI.gaussian(X, y, lam, approximate_inverse=None)
    R.fit()
    summaryR = R.summary(truth=truth[R.active], dispersion=dispersion, compute_intervals=True, level=1-alpha)
    summaryR0 = R.summary(dispersion=dispersion, compute_intervals=False)

    instance_hash = hashlib.md5()
    instance_hash.update(X.tobytes())
    instance_hash.update(y.tobytes())
    instance_hash.update(truth.tobytes())
    instance_id = instance_hash.hexdigest()

    if summaryR is not None:
        liu_pivots = summaryR['pval']
        liu_pvalues = summaryR0['pval']
        liu_lower = summaryR['lower_confidence']
        liu_upper = summaryR['upper_confidence']
        liu_lengths = liu_upper - liu_lower
        liu_covered = [(l < t) * (t < u) for l, u, t in zip(liu_lower, liu_upper, truth[R.active])]
    else:
        liu_pivots = liu_pvalues = liu_lower = liu_upper = liu_lengths = liu_covered = []

    return pd.DataFrame({'liu_pivot':liu_pivots,
                         'liu_pvalue':liu_pvalues,
                         'liu_length':liu_lengths,
                         'liu_upper':liu_upper,
                         'liu_lower':liu_lower,
                         'liu_coverage':liu_covered,
                         'target':truth[R.active],
                         'id':[instance_hash]*len(liu_pivots)})
