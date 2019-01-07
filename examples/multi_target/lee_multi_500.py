import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import lasso

from learn_selection.core import (infer_general_target,
                                  split_sampler, 
                                  normal_sampler,
                                  logit_fit,
                                  repeat_selection,
                                  probit_fit)
from learn_selection.keras_fit import keras_fit
from learn_selection.learners import sparse_mixture_learner

def simulate(n=2000, p=500, s=20, signal=(3 / np.sqrt(2000), 4 / np.sqrt(2000)), sigma=2, alpha=0.1, B=10000):

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
    print(np.linalg.norm(truth))

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)

    def meta_algorithm(XTX, XTXi, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.
        noisy_S = sampler(scale=scale)
        loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve(max_its=300, tol=1.e-10)
        success += soln != 0
        return tuple(sorted(np.nonzero(success)[0]))

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, lam)

    # run selection algorithm

    success_params = (1, 1)

    observed_tuple = selection_algorithm(smooth_sampler)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths, pvalues = [], [], [], []
    lower, upper = [], []
    naive_pvalues, naive_pivots, naive_covered, naive_lengths =  [], [], [], []

    L = lasso.gaussian(X, y, lam, sigma=np.sqrt(dispersion))
    L.fit()
    summaryL = None

    targets = []

    observed_list = list(observed_tuple)

    howmany = len(observed_list)
    if len(observed_tuple) > 0:
        print(observed_tuple)
        Xpi = np.linalg.pinv(X[:,observed_list])
        final_target = Xpi.dot(X.dot(truth))
        summaryL0 = L.summary(compute_intervals=False)
        summaryL = L.summary(truth=final_target, compute_intervals=True, level=1-alpha)

        observed_target = Xpi.dot(y)

        sel_dispersion = dispersion # np.linalg.norm(y - X[:,observed_list].dot(Xpi.dot(y)))**2 / (n - len(observed_list))
        print(sel_dispersion, dispersion)
        cov_target = Xpi.dot(Xpi.T) * sel_dispersion
        cross_cov = X.T.dot(Xpi.T) * sel_dispersion

        results = infer_general_target(selection_algorithm,
                                       observed_tuple,
                                       smooth_sampler,
                                       observed_target[:howmany],
                                       cross_cov[:,:howmany],
                                       cov_target[:howmany][:,:howmany],
                                       hypothesis=final_target[:howmany],
                                       fit_probability=keras_fit,
                                       fit_args={'epochs':30, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'},
                                       alpha=alpha,
                                       B=B,
                                       learner_klass=sparse_mixture_learner)

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

        target_sd = np.sqrt(np.diag(cov_target))
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)
        naive_lower, naive_upper = naive_interval

        naive_pivots = (1 - ndist.cdf((observed_target - final_target) / target_sd))
        naive_pivots = 2 * np.minimum(naive_pivots, 1 - naive_pivots)

        naive_pvalues = (1 - ndist.cdf(observed_target / target_sd))
        naive_pvalues = 2 * np.minimum(naive_pvalues, 1 - naive_pvalues)

        naive_covered = (naive_interval[0] < final_target) * (naive_interval[1] > final_target)
        naive_lengths = naive_interval[1] - naive_interval[0]

    if summaryL is not None:
        lee_pivots = summaryL['pval']
        lee_pvalues = summaryL0['pval']
        lee_lower = summaryL['lower_confidence']
        lee_upper = summaryL['upper_confidence']
        lee_lengths = lee_upper - lee_lower
        lee_covered = [(l < t) * (t < u) for l, u, t in zip(lee_lower, lee_upper, final_target)]
    else:
        lee_pivots = lee_pvalues = lee_lower = lee_upper = lee_lengths = lee_covered = []

    if len(pvalues) > 0:
        return pd.DataFrame({'pivot':pivots,
                             'pvalue':pvalues,
                             'coverage':covered,
                             'length':lengths,
                             'naive_pivot':naive_pivots[:howmany],
                             'naive_coverage':naive_covered[:howmany],
                             'naive_length':naive_lengths[:howmany],
                             'lee_pivot':lee_pivots[:howmany],
                             'lee_pvalue':lee_pvalues[:howmany],
                             'lee_length':lee_lengths[:howmany],
                             'lee_upper':lee_upper[:howmany],
                             'lee_lower':lee_lower[:howmany],
                             'upper':upper,
                             'lower':lower,
                             'lee_coverage':lee_covered[:howmany],
#                             'target':final_target,
                             })


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    for i in range(500):
        df = simulate(B=10000)
        csvfile = 'lee_multi_500.csv'

        if df is not None:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass

            df.to_csv(csvfile)

            if len(df['pivot']) > 0:

                print("selective:", np.mean(df['pivot']), np.std(df['pivot']), np.mean(df['length']), np.std(df['length']), np.mean(df['coverage']))
                print("lee:", np.mean(df['lee_pivot']), np.std(df['lee_pivot']), np.mean(df['lee_length']), np.std(df['lee_length']), np.mean(df['lee_coverage']))
                print("naive:", np.mean(df['naive_pivot']), np.std(df['naive_pivot']), np.mean(df['naive_length']), np.std(df['naive_length']), np.mean(df['naive_coverage']))

                print("len ratio selective divided by naive:", np.mean(np.array(df['length']) / np.array(df['naive_length'])))
                print("len ratio selective divided by lee:", np.mean(np.array(df['length']) / np.array(df['lee_length'])))

                plt.clf()
                U = np.linspace(0, 1, 101)
                plt.plot(U, sm.distributions.ECDF(df['pivot'])(U), 'r', label='Selective', linewidth=3)
                plt.plot(U, sm.distributions.ECDF(df['naive_pivot'])(U), 'b', label='Naive', linewidth=3)
                plt.plot(U, sm.distributions.ECDF(df['lee_pivot'][~np.isnan(df['lee_pivot'])])(U), 'g', label='Lee', linewidth=3)
                plt.legend()
                plt.plot([0,1], [0,1], 'k--', linewidth=2)
                plt.savefig(csvfile[:-4] + '.pdf')

                plt.clf()
                plt.scatter(df['naive_length'], df['length'])
                plt.scatter(df['naive_length'], df['lee_length'])
                plt.savefig(csvfile[:-4] + '_lengths.pdf')

                df.to_csv(csvfile, index=False)
            
