import hashlib, warnings

import numpy as np
import pandas as pd
from scipy.stats import norm as normal_dbn

from ..algorithms.lasso import ROSI, lasso

from .core import (infer_full_target,
                   infer_general_target,
                   repeat_selection,
                   gbm_fit_sk)
from .learners import mixture_learner

def full_model_inference(X, 
                         y,
                         truth,
                         selection_algorithm,
                         sampler,
                         success_params=(1, 1),
                         fit_probability=gbm_fit_sk,
                         fit_args={'n_estimators':500},
                         alpha=0.1,
                         B=2000,
                         naive=True,
                         learner_klass=mixture_learner,
                         features=None,
                         how_many=None):

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
    if features is not None:
        observed_set = observed_set.intersection(features)
    observed_list = sorted(observed_set)
    if len(observed_list) > 0:

        if how_many is None:
            how_many = len(observed_list)
        observed_list = observed_list[:how_many]

        # find the target, based on the observed outcome

        (pivots, 
         covered, 
         lengths, 
         pvalues,
         lower,
         upper) = [], [], [], [], [], []

        targets = []
        true_target = truth[observed_list]

        results = infer_full_target(selection_algorithm,
                                    observed_set,
                                    observed_list,
                                    sampler,
                                    dispersion,
                                    hypothesis=true_target,
                                    fit_probability=fit_probability,
                                    fit_args=fit_args,
                                    success_params=success_params,
                                    alpha=alpha,
                                    B=B,
                                    learner_klass=learner_klass)

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

        if len(pvalues) > 0:
            df = pd.DataFrame({'pivot':pivots,
                               'pvalue':pvalues,
                               'coverage':covered,
                               'length':lengths,
                               'upper':upper,
                               'lower':lower,
                               'id':[instance_id]*len(pvalues),
                               'nfeature':X.shape[1],
                               'alpha':alpha,
                               'nsample':X.shape[0],
                               'target':true_target,
                               'variable':observed_list,
                               'B':[B]*len(pvalues)})
            if naive:
                naive_df = naive_full_model_inference(X, 
                                                      y,
                                                      dispersion,
                                                      truth,
                                                      observed_set,
                                                      alpha,
                                                      how_many=how_many)
                df = pd.merge(df, naive_df, on='variable')
            return df

def split_full_model_inference(X,
                               y,
                               idx,
                               dispersion,
                               truth,
                               observed_set,
                               alpha=0.1,
                               how_many=None):

    n, p = X.shape

    stage_2 = sorted(set(range(n)).difference(idx))
    X2 = X[stage_2]
    y2 = y[stage_2]

    XTXi_2 = np.linalg.inv(X2.T.dot(X2))
    resid2 = y2 - X2.dot(XTXi_2.dot(X2.T.dot(y2)))
    dispersion_2 = np.linalg.norm(resid2)**2 / (X2.shape[0] - X2.shape[1])

    split_df = naive_full_model_inference(X2,
                                          y2,
                                          dispersion_2,
                                          truth,
                                          observed_set,
                                          alpha=alpha)

    split_df = split_df.rename(columns=dict([(v,
                                              v.replace('naive', 'split'))
                                             for v in split_df.columns]))

    for n in split_df.columns:
        if 'bonferroni' in n:
            split_df = split_df.drop(n, axis=1)

    return split_df

def split_partial_model_inference(X,
                                  y,
                                  idx,
                                  dispersion,
                                  truth,
                                  observed_set,
                                  alpha=0.1,
                                  how_many=None):

    n, p = X.shape

    stage_2 = sorted(set(range(n)).difference(idx))
    X2 = X[stage_2]
    y2 = y[stage_2]

    XTXi_2 = np.linalg.inv(X2.T.dot(X2))
    resid2 = y2 - X2.dot(XTXi_2.dot(X2.T.dot(y2)))
    dispersion_2 = np.linalg.norm(resid2)**2 / (X2.shape[0] - X2.shape[1])

    split_df = naive_partial_model_inference(X2,
                                             y2,
                                             dispersion_2,
                                             truth,
                                             observed_set,
                                             alpha=alpha)

    split_df = split_df.rename(columns=dict([(v,
                                              v.replace('naive', 'split'))
                                             for v in split_df.columns]))
    return split_df


def naive_full_model_inference(X,
                               y,
                               dispersion,
                               truth,
                               observed_set,
                               alpha=0.1,
                               how_many=None):
    
    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)

    (naive_pvalues, 
     naive_pivots, 
     naive_covered, 
     naive_lengths, 
     naive_upper, 
     naive_lower) =  [], [], [], [], [], []

    (bonferroni_pvalues, 
     bonferroni_covered, 
     bonferroni_lengths, 
     bonferroni_upper, 
     bonferroni_lower) =  [], [], [], [], []

    observed_list = sorted(observed_set)
    if how_many is None:
        how_many = len(observed_list)
    observed_list = observed_list[:how_many]

    for idx in observed_list:
        true_target = truth[idx]
        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))

        # uncorrected

        quantile = normal_dbn.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, 
                          observed_target + quantile * target_sd)
        naive_upper.append(naive_interval[1])
        naive_lower.append(naive_interval[0])
        naive_pivot = (1 - normal_dbn.cdf((observed_target - true_target) / target_sd))
        naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pivots.append(naive_pivot)

        naive_pvalue = (1 - normal_dbn.cdf(observed_target / target_sd))
        naive_pvalue = 2 * min(naive_pvalue, 1 - naive_pvalue)
        naive_pvalues.append(naive_pvalue)

        naive_covered.append((naive_interval[0] < true_target) * (naive_interval[1] > true_target))
        naive_lengths.append(naive_interval[1] - naive_interval[0])

        # Bonferroni

        nfeature = X.shape[1]
        quantile = normal_dbn.ppf(1 - 0.5 * alpha / nfeature)
        bonferroni_interval = (observed_target - quantile * target_sd, 
                               observed_target + quantile * target_sd)
        bonferroni_upper.append(bonferroni_interval[1])
        bonferroni_lower.append(bonferroni_interval[0])

        bonferroni_pvalue = min(1, nfeature * naive_pvalue)
        bonferroni_pvalues.append(bonferroni_pvalue)

        bonferroni_covered.append((bonferroni_interval[0] < true_target) * (bonferroni_interval[1] > true_target))
        bonferroni_lengths.append(bonferroni_interval[1] - bonferroni_interval[0])


    return pd.DataFrame({'naive_pivot':naive_pivots,
                         'naive_pvalue':naive_pvalues,
                         'naive_coverage':naive_covered,
                         'naive_length':naive_lengths,
                         'naive_upper':naive_upper,
                         'naive_lower':naive_lower,
                         'bonferroni_pvalue':bonferroni_pvalues,
                         'bonferroni_coverage':bonferroni_covered,
                         'bonferroni_length':bonferroni_lengths,
                         'bonferroni_upper':bonferroni_upper,
                         'bonferroni_lower':bonferroni_lower,
                         'variable':observed_list,
                         })

def partial_model_inference(X, 
                            y,
                            truth,
                            selection_algorithm,
                            sampler,
                            success_params=(1, 1),
                            fit_probability=gbm_fit_sk,
                            fit_args={'n_estimators':500},
                            alpha=0.1,
                            B=2000,
                            naive=True,
                            learner_klass=mixture_learner):

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

    observed_tuple = selection_algorithm(sampler)

    (pivots, 
     covered, 
     lengths, 
     pvalues,
     lower,
     upper) = [], [], [], [], [], []

    targets = []


    if len(observed_tuple) > 0:

        Xpi = np.linalg.pinv(X[:, list(observed_tuple)])
        final_target = Xpi.dot(X.dot(truth))
        observed_target = Xpi.dot(y)

        target_cov = Xpi.dot(Xpi.T) * dispersion
        cross_cov = X.T.dot(Xpi.T) * dispersion

        learner = learner_klass(selection_algorithm, 
                                observed_tuple,
                                sampler, 
                                observed_target,
                                target_cov,
                                cross_cov)

        results = infer_general_target(observed_tuple,
                                       observed_target,
                                       target_cov,
                                       learner,
                                       hypothesis=final_target,
                                       fit_probability=fit_probability,
                                       fit_args=fit_args,
                                       alpha=alpha,
                                       B=B)

        for result, true_target in zip(results, final_target):
            (pivot, 
             interval,
             pvalue,
             _) = result
            
            pvalues.append(pvalue)
            pivots.append(pivot)
            covered.append((interval[0] < true_target) * (interval[1] > true_target))
            lengths.append(interval[1] - interval[0])
            lower.append(interval[0])
            upper.append(interval[1])

    if len(observed_tuple) > 0:

        df = pd.DataFrame({'pivot':pivots,
                           'pvalue':pvalues,
                           'coverage':covered,
                           'length':lengths,
                           'nfeature':X.shape[1],
                           'alpha':alpha,
                           'nsample':X.shape[0],
                           'upper':upper,
                           'lower':lower,
                           'target':final_target,
                           'variable':list(observed_tuple),
                           'id':[instance_id]*len(pivots),
                           })
        if naive:
            naive_df = naive_partial_model_inference(X,
                                                     y,
                                                     dispersion,
                                                     truth,
                                                     observed_tuple,
                                                     alpha=alpha)
            df = pd.merge(df, naive_df, on='variable')

        return df

def naive_partial_model_inference(X,
                                  y,
                                  dispersion,
                                  truth,
                                  observed_set,
                                  alpha=0.1):

    if len(observed_set) > 0:

        observed_list = sorted(observed_set)
        Xpi = np.linalg.pinv(X[:,observed_list])
        final_target = Xpi.dot(X.dot(truth))

        observed_target = Xpi.dot(y)

        target_cov = Xpi.dot(Xpi.T) * dispersion
        cross_cov = X.T.dot(Xpi.T) * dispersion

        target_sd = np.sqrt(np.diag(target_cov))
        quantile = normal_dbn.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)
        naive_lower, naive_upper = naive_interval

        naive_pivots = (1 - normal_dbn.cdf((observed_target - final_target) / target_sd))
        naive_pivots = 2 * np.minimum(naive_pivots, 1 - naive_pivots)

        naive_pvalues = (1 - normal_dbn.cdf(observed_target / target_sd))
        naive_pvalues = 2 * np.minimum(naive_pvalues, 1 - naive_pvalues)

        naive_covered = (naive_interval[0] < final_target) * (naive_interval[1] > final_target)
        naive_lengths = naive_interval[1] - naive_interval[0]

        return pd.DataFrame({'naive_pivot':naive_pivots,
                             'naive_coverage':naive_covered,
                             'naive_length':naive_lengths,
                             'nfeature':X.shape[1],
                             'naive_lower':naive_lower,
                             'target':final_target,
                             'variable':observed_list
                             })

def lee_inference(X,
                  y,
                  lam,
                  dispersion,
                  truth,
                  alpha=0.1):

    L = lasso.gaussian(X, y, lam, sigma=np.sqrt(dispersion))
    L.fit()

    Xpi = np.linalg.pinv(X[:,L.active])
    final_target = Xpi.dot(X.dot(truth))

    summaryL0 = L.summary(compute_intervals=False)

    lee_pvalues = summaryL0['pval']
    lee_lower = summaryL0['lower_confidence']
    lee_upper = summaryL0['upper_confidence']
    lee_lengths = lee_upper - lee_lower
    lee_pivots = lee_pvalues * np.nan
    lee_covered = [(l < t) * (t < u) for l, u, t in zip(lee_lower, lee_upper, final_target)]

    return pd.DataFrame({'lee_pivot':lee_pivots,
                         'lee_pvalue':lee_pvalues,
                         'lee_length':lee_lengths,
                         'lee_upper':lee_upper,
                         'lee_lower':lee_lower,
                         'lee_coverage':lee_covered,
                         'variable':summaryL0['variable']})

try:
    import matplotlib.pyplot as plt

    def pivot_plot(df, 
                   outbase,
                   figsize=(8,8)):

        print("selective:", np.mean(df['pivot']), np.std(df['pivot']), np.mean(df['length']), np.std(df['length']), np.mean(df['coverage']))
        print("naive:", np.mean(df['naive_pivot']), np.std(df['naive_pivot']), np.mean(df['naive_length']), np.std(df['naive_length']), np.mean(df['naive_coverage']))

        print("len ratio selective divided by naive:", np.mean(np.array(df['length']) / np.array(df['naive_length'])))

        f = plt.figure(num=1, figsize=figsize)
        plt.clf()
        U = np.linspace(0, 1, 101)
        plt.plot(U, sm.distributions.ECDF(df['pivot'])(U), 'b', label='Selective', linewidth=3)
        plt.plot(U, sm.distributions.ECDF(df['naive_pivot'])(U), 'r', label='Naive', linewidth=3)
        plt.legend(fontsize=15)
        plt.plot([0,1], [0,1], 'k--', linewidth=2)
        plt.savefig(outbase + '.pdf')
        pivot_ax = plt.gca()
        pivot_ax.set_ylabel(r'P(pivot < t)')
        pivot_ax.set_xlabel(r't')

        return pivot_ax

except ImportError:
    warnings.warn('matplotlib not importable, pivot_plot will not be available')

def liu_inference(X,
                  y,
                  lam,
                  dispersion,
                  truth,
                  alpha=0.1,
                  approximate_inverse=None):

    R = ROSI.gaussian(X, y, lam, approximate_inverse=approximate_inverse)
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
        variable = summaryR['variable']
        liu_lengths = liu_upper - liu_lower
        liu_covered = [(l < t) * (t < u) for l, u, t in zip(liu_lower, liu_upper, truth[R.active])]
    else:
        variable = liu_pivots = liu_pvalues = liu_lower = liu_upper = liu_lengths = liu_covered = []

    return pd.DataFrame({'liu_pivot':liu_pivots,
                         'liu_pvalue':liu_pvalues,
                         'liu_length':liu_lengths,
                         'liu_upper':liu_upper,
                         'liu_lower':liu_lower,
                         'liu_coverage':liu_covered,
                         'liu_upper':liu_upper,
                         'liu_lower':liu_lower,
                         'target':truth[R.active],
                         'id':[instance_id]*len(liu_pivots),
                         'variable':variable})


try:
    import statsmodels.api as sm

    def pvalue_plot(df, 
                    outbase,
                    figsize=(8, 8),
                    naive=True,
                    split=False,
                    bonferroni=False):

        print("selective:", np.mean(df['pvalue']), np.std(df['pvalue']), np.mean(df['length']), np.std(df['length']), np.mean(df['coverage']))

        if naive:
            print("naive:", np.mean(df['naive_length']), np.std(df['naive_length']), np.mean(df['naive_coverage']))
            print("len ratio selective divided by naive:", np.mean(np.array(df['length']) / np.array(df['naive_length'])))

        if split:
            print("split:", np.mean(df['split_length']), np.std(df['split_length']), np.mean(df['split_coverage']))
            print("len ratio selective divided by split:", np.mean(np.array(df['length']) / np.array(df['split_length'])))

        if bonferroni:
            print("bonferroni:", np.mean(df['bonferroni_length']), np.std(df['bonferroni_length']), np.mean(df['bonferroni_coverage']))
            print("len ratio selective divided by bonferroni:", np.mean(np.array(df['length']) / np.array(df['bonferroni_length'])))

        f = plt.figure(figsize=figsize)
        plt.clf()
        U = np.linspace(0, 1, 101)
        non_null = df['target'] != 0
        null = ~non_null
        if non_null.sum():
            plt.plot(U, sm.distributions.ECDF(df['pvalue'][non_null])(U), 'b', label='Learned', linewidth=3)
        if null.sum():
            plt.plot(U, sm.distributions.ECDF(df['pvalue'][null])(U), 'b--', linewidth=3)

        if naive:
            if non_null.sum():
                plt.plot(U, sm.distributions.ECDF(df['naive_pvalue'][non_null])(U), 'r', label='Naive', linewidth=3)
            if null.sum():
                plt.plot(U, sm.distributions.ECDF(df['naive_pvalue'][null])(U), 'r--', linewidth=3)

        if split:
            if non_null.sum():
                plt.plot(U, sm.distributions.ECDF(df['split_pvalue'][non_null])(U), color='gray', label='Split', linewidth=3)
            if null.sum():
                plt.plot(U, sm.distributions.ECDF(df['split_pvalue'][null])(U), linestyle='dashed', color='gray', linewidth=3)

        if bonferroni:
            if non_null.sum():
                plt.plot(U, sm.distributions.ECDF(df['bonferroni_pvalue'][non_null])(U), color='purple', label='Bonferroni', linewidth=3)
            if null.sum():
                plt.plot(U, sm.distributions.ECDF(df['bonferroni_pvalue'][null])(U), linestyle='dashed', color='purple', linewidth=3)

        plt.legend(fontsize=15)
        plt.plot([0,1], [0,1], 'k--', linewidth=3)

        pvalue_ax = plt.gca()
        pvalue_ax.set_ylabel(r'ECDF(pvalue)', fontsize=20)
        pvalue_ax.set_xlabel(r'pvalue', fontsize=20)

        plt.savefig(outbase + '_pvalues.pdf')
        plt.savefig(outbase + '_pvalues.png', dpi=300)

        return pvalue_ax

    def pivot_plot_new(df,
                       outbase,
                       palette = {'Learned': 'b',
                                  'Naive': 'r',
                                  'Bonferroni': 'gray',
                                  'Lee':'gray',
                                  'Strawman':'gray'},
                       figsize=(8, 8), straw=False):

        f = plt.figure(figsize=figsize)
        new_df = pd.DataFrame({'Learned': df['pivot'],
                               'Naive': df['naive_pivot']})
        if straw:
            new_df = pd.DataFrame({'Learned': new_df['Learned'],
                                   'Strawman': new_df['Naive']})
        U = np.linspace(0, 1, 101)
        ax = f.gca()
        for k in new_df.keys():
            plt.plot(U, sm.distributions.ECDF(new_df[k])(U), color=palette[k], label=k, linewidth=5)
        plt.plot([0,1], [0,1], 'k--', linewidth=3)
        ax.set_xlabel('pivot', fontsize=20)
        ax.set_ylabel('ECDF(pivot)', fontsize=20)
        ax.legend(fontsize=15)

        pngfile = outbase + '_pivot.png'
        plt.savefig(pngfile, dpi=300)

        return ax, f, pngfile, df, new_df

except:
    warnings.warn('statsmodels not importable, `pvalue_plot` and `pvalue_plot_new` unavaliable')

# Some plotting functions

try:
    import seaborn as sns

    def interval_plot(df, 
                      outbase,
                      palette = {'Learned': 'b',
                                 'Naive': 'r',
                                 'Bonferroni': 'purple',
                                 'Split':'gray'},
                      figsize=(8, 8), 
                      naive=True,
                      bonferroni=True,
                      split=False,
                      xlim=None):

        f = plt.figure(figsize=figsize)
        new_df = pd.DataFrame({'Learned': df['length'],
                               'Naive': df['naive_length']})
        if bonferroni:
            new_df['Bonferroni'] = df['bonferroni_length']
        ax = f.gca()

        if split:
            new_df['Split'] = df['split_length']
        for k in new_df.keys():
            l = new_df[k]
            l = l[~np.isnan(l)]
            sns.distplot(l, ax=ax, color=palette[k], label=k)
        ax.set_xlabel('Interval length', fontsize=20)
        ax.set_yticks([])
        ax.legend(fontsize=15)
        if xlim is not None:
            ax.set_xlim(xlim)

        pngfile = outbase + '_intervals.png'
        plt.savefig(pngfile, dpi=300)
        plt.savefig(outbase + '_intervals.pdf')

        return ax, f, pngfile, df, new_df

except ImportError:
    warnings.warn('seaborn not found, `interval_plot` will not be available')
