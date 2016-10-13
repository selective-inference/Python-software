"""
special column names:
mle -- pivot at unpenalized MLE
truth -- pivot at true parameter
pvalue -- tests of H0 for each variable
count -- how many runs (including last one) until success
active -- was variable truly active
naive_pvalue --
cover --
naive_cover --
"""
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot, uniform
import statsmodels.api as sm

def collect_multiple_runs(test_fn, columns, nrun, summary_fn, *args, **kwargs):
    """
    Assumes a wait_for_return_value test...
    """
    dfs = []
    for i in range(nrun):
        count, result = test_fn(*args, **kwargs)

        df_i = pd.DataFrame(index=np.arange(len(np.atleast_1d(result[0]))),
                            columns=columns + ['count', 'run'])
        print(result[0])
        print(len(df_i))

        df_i.loc[:,'count'] = count
        df_i.loc[:,'run'] = i

        for col, v in zip(columns, result):
            df_i.loc[:,col] = np.atleast_1d(v)
        dfs.append(df_i)
        if summary_fn is not None:
            summary_fn(pd.concat(dfs))
    return pd.concat(dfs)

def pvalue_plot(multiple_results, screening=False, fig=None, colors=['r','g']):
    """
    Extract pvalues and group by
    null and alternative.
    """

    P0 = multiple_results['pvalue'][~multiple_results['active']]
    P0 = P0[~pd.isnull(P0)]
    PA = multiple_results['pvalue'][multiple_results['active']]
    PA = PA[~pd.isnull(PA)]

    if fig is None:
        fig = plt.figure()
    ax = fig.gca()

    fig.suptitle('Null and alternative p-values')

    grid = np.linspace(0, 1, 51)

    if len(P0) > 0:
        ecdf0 = sm.distributions.ECDF(P0)
        F0 = ecdf0(grid)
        ax.plot(grid, F0, '--o', c=colors[0], lw=2, label=r'$H_0$')
    if len(PA) > 0:
        ecdfA = sm.distributions.ECDF(PA)
        FA = ecdfA(grid)
        ax.plot(grid, FA, '--o', c=colors[1], lw=2, label=r'$H_A$')

    ax.plot([0, 1], [0, 1], 'k-', lw=2)
    ax.legend(loc='lower right')

    if screening:
        screen = 1. / np.mean(multiple_results.loc[multiple_results.index == 0,'count'])
        ax.set_title('Screening: %0.2f' % screen)
    return fig

def naive_pvalue_plot(multiple_results, screening=False, fig=None, colors=['r', 'g']):
    """
    Extract naive pvalues and group by
    null and alternative.
    """

    P0 = multiple_results['naive_pvalue'][~multiple_results['active']]
    P0 = P0[~pd.isnull(P0)]
    PA = multiple_results['naive_pvalue'][multiple_results['active']]
    PA = PA[~pd.isnull(PA)]

    if fig is None:
        fig = plt.figure()
    ax = fig.gca()

    fig.suptitle('Null and alternative p-values')

    grid = np.linspace(0, 1, 51)

    if len(P0) > 0:
        ecdf0 = sm.distributions.ECDF(P0)
        F0 = ecdf0(grid)
        ax.plot(grid, F0, '--o', c=colors[0], lw=2, label=r'$H_0$ naive')
    if len(PA) > 0:
        ecdfA = sm.distributions.ECDF(PA)
        FA = ecdfA(grid)
        ax.plot(grid, FA, '--o', c=colors[1], lw=2, label=r'$H_A$ naive')

    ax.plot([0, 1], [0, 1], 'k-', lw=2)
    ax.legend(loc='lower right')

    if screening:
        screen = 1. / np.mean(multiple_results.loc[multiple_results.index == 0,'count'])
        ax.set_title('Screening: %0.2f' % screen)

    return fig

def pivot_plot(multiple_results, coverage=True, color='b', label=None, fig=None):
    """
    Extract pivots at truth and mle.
    """

    if fig is None:
        fig, _ = plt.subplots(nrows=1, ncols=2)
    plot_pvalues_clt, plot_pvalues_boot = fig.axes

    ecdf_clt = sm.distributions.ECDF(multiple_results['pivots_clt'])
    G = np.linspace(0, 1)
    F_MLE = ecdf_clt(G)
    #print(color)
    plot_pvalues_clt.plot(G, F_MLE, '-o', c=color, lw=2, label=label)
    plot_pvalues_clt.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues_clt.set_title("Pivots based on plugin CLT")
    plot_pvalues_clt.set_xlim([0, 1])
    plot_pvalues_clt.set_ylim([0, 1])
    #plot_pvalues_clt.legend(loc='lower right')

    ecdf_boot = sm.distributions.ECDF(multiple_results['pivots_boot'])
    F_true = ecdf_boot(G)
    plot_pvalues_boot.plot(G, F_true, '-o', c=color, lw=2, label=label)
    plot_pvalues_boot.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues_boot.set_title("Bootstrapped Pivots")
    plot_pvalues_boot.set_xlim([0, 1])
    plot_pvalues_boot.set_ylim([0, 1])
    #plot_pvalues_boot.legend(loc='lower right')

    if coverage:
        if 'covered_naive' in multiple_results.columns:
            fig.suptitle('CLT Coverage: %0.2f, Boot: %0.2f, Naive: %0.2f' % (np.mean(multiple_results['covered_clt']),
                            np.mean(multiple_results['cover_boot']), np.mean(multiple_results['covered_naive'])))
        else:
            fig.suptitle('Coverage CLT: %0.2f, Bootstrap: %0.2f' % (np.mean(multiple_results['covered_clt']),
                                                            np.mean(multiple_results['covered_boot'])))
    fig.show()
    return fig

def compute_pivots(multiple_results):
    pivot_summary = {}
    if 'pivots_clt' in multiple_results.columns:
        pivots_clt = multiple_results['pivots_clt']
        pivot_summary['pivots_clt'] = {'CLT pivots (mean, SD, type I):': (np.mean(pivots_clt), np.std(pivots_clt), np.mean(pivots_clt < 0.05))}
    if 'pivots_boot' in multiple_results.columns:
        pivots_boot = multiple_results['pivots_boot']
        pivot_summary['pivots_boot'] = {'Bootstrap pivots (mean, SD, type I):': (np.mean(pivots_boot), np.std(pivots_boot), np.mean(pivots_boot < 0.05))}
    return pivot_summary


def compute_coverage(multiple_results):
    result = {}
    if 'covered_naive' in multiple_results.columns:
        result['naive coverage'] = np.mean(multiple_results['covered_naive'])
    if 'covered_clt' in multiple_results.columns:
        result['selective coverage (CLT)'] = np.mean(multiple_results['covered_clt'])
    if 'covered_boot' in multiple_results.columns:
        result['selective coverage (Bootstrap)'] = np.mean(multiple_results['covered_boot'])
    return result


def compute_lengths(multiple_results):
    result = {}
    if 'ci_length_clt' in multiple_results.columns:
        print(multiple_results['ci_length_clt'])
        result['ci_length_clt'] = np.mean(multiple_results['ci_length_clt'])
    if 'ci_length_boot' in multiple_results.columns:
        result['ci_length_boot'] = np.mean(multiple_results['ci_length_boot'])
    if 'ci_length_clt' in multiple_results.columns:
        result['ci_length_split'] = np.mean(multiple_results['ci_length_split'])

    return result


def compute_screening(multiple_results):
    return {'screening:': 1. / np.mean(multiple_results.loc[multiple_results.index == 0,'count'])}

def summarize_all(multiple_results):
    result = {}
    result.update(compute_pivots(multiple_results))
    result.update(compute_coverage(multiple_results))
    result.update(compute_screening(multiple_results))
    result.update(compute_lengths(multiple_results))
    print(result)
reports = {}

# if __name__ == "__main__":

#     from selection.randomized.tests.test_split import test_split

#     pvalue_plot(collect_multiple_runs(test_split,
#                                       ['mle', 'truth', 'pvalue', 'cover', 'naive_cover', 'active'],
#                                       3,
#                                       None,
#                                       bootstrap=True),
#                 screening=True)