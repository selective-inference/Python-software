import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import lasso, ROSI
from learn_selection.knockoffs import lasso_glmnet

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(knockoff); library(glmnet)')
from rpy2 import rinterface

from learn_selection.core import (infer_general_target,
                                  split_sampler,
                                  normal_sampler,
                                  logit_fit,
                                  repeat_selection,
                                  probit_fit)

def simulate(n=100, p=50, s=5, signal=(0, 0), sigma=2, alpha=0.1, glmnet=True):

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

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    if not glmnet:
        def meta_algorithm(XTX, XTXi, lam, sampler):

            p = XTX.shape[0]
            success = np.zeros(p)

            loss = rr.quadratic_loss((p,), Q=XTX)
            pen = rr.l1norm(p, lagrange=lam)

            scale = 0.5
            noisy_S = sampler(scale=0)
            loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
            problem = rr.simple_problem(loss, pen)
            soln = problem.solve(max_its=50, tol=1.e-6)
            success += soln != 0
            return set(np.nonzero(success)[0])

        XTX = X.T.dot(X)
        XTXi = np.linalg.inv(XTX)
        resid = y - X.dot(XTXi.dot(X.T.dot(y)))
        dispersion = np.linalg.norm(resid)**2 / (n-p)

        selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, 4. * np.sqrt(n))

    else:

        def meta_algorithm(X, XTXi, resid, sampler):

            S = sampler(scale=0.) # deterministic with scale=0
            ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
            G = lasso_glmnet(X, ynew, *[None]*4)
            select = G.select()
            return set(list(select[0]))

        XTX = X.T.dot(X)
        XTXi = np.linalg.inv(XTX)
        resid = y - X.dot(XTXi.dot(X.T.dot(y)))
        dispersion = np.linalg.norm(resid)**2 / (n-p)

        selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

    # run selection algorithm

    success_params = (1, 1)

    observed_set = repeat_selection(selection_algorithm, splitting_sampler, *success_params)

    # find the target, based on the observed outcome

    # we just take the first target  

    pvalues, covered, lengths = [], [], []
    naive_pvalues, naive_covered, naive_lengths =  [], [], []
    lee_pvalues, liu_pvalues = [], []

    observed_list = sorted(observed_set)

    if len(observed_list) > 0:
        idx = observed_list[0]
        print("variable: ", idx, "total selected: ", len(observed_set))

        linfunc = np.linalg.pinv(X[:,observed_list])[0]
        true_target = linfunc.dot(X.dot(truth))
        observed_target = np.array([linfunc.dot(y)])
        cov_target = np.array([[np.linalg.norm(linfunc)**2 * dispersion]])
        cross_cov = X.T.dot(linfunc).reshape((-1,1)) * dispersion

        (pivot, 
         interval) = infer_general_target(selection_algorithm,
                                          observed_set,
                                          splitting_sampler,
                                          observed_target,
                                          cross_cov,
                                          cov_target,
                                          hypothesis=[true_target],
                                          fit_probability=probit_fit,
                                          alpha=alpha,
                                          B=2000)[:2]

        pvalues.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(cov_target[0, 0])
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)
        naive_pivot = (1 - ndist.cdf((observed_target - true_target) / target_sd))
        naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pvalues.append(naive_pivot)

        naive_covered.append((naive_interval[0] < true_target) * (naive_interval[1] > true_target))
        naive_lengths.append(naive_interval[1] - naive_interval[0])

        # lee and rosi

        numpy2ri.activate()
        rpy.r.assign('X', X)
        rpy.r.assign('Y', y)
        rpy.r('X = as.matrix(X)')
        rpy.r('Y = as.numeric(Y)')
        rpy.r('cvG = cv.glmnet(X, Y, intercept=FALSE, standardize=FALSE)')
        lam = rpy.r('cvG$lambda.min')[0]
        numpy2ri.deactivate()

        try:
            n, p = X.shape
            L = lasso.gaussian(X, y, n * lam)
            L.fit()
            lee_pvalues = [np.array(L.summary()['pval'])[0]]
        except:
            lee_pvalues = [np.nan]

        try:
            R = ROSI.gaussian(X, y, n * lam, sigma=np.sqrt(dispersion), approximate_inverse=None)
            R.fit()
            pv = np.array(R.summary()['pval'])[0]
            pv = 2 * min(pv, 1 - pv)
            liu_pvalues = [pv]
        except:
            liu_pvalues = [np.nan]


    return pvalues, covered, lengths, naive_pvalues, naive_covered, naive_lengths, lee_pvalues, liu_pvalues


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(500):
        p, cover, l, naive_p, naive_covered, naive_l, lee, liu = simulate()

        csvfile = 'lasso_exact_variable.csv'

        if i % 2 == 1 and i > 0:

            df = pd.DataFrame({'pivot':p,
                               'naive_pivot':naive_p,
                               'coverage':cover,
                               'naive_coverage':naive_covered,
                               'length':l,
                               'naive_length':naive_l,
                               'lee_pivots':lee,
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
                plt.plot(U, sm.distributions.ECDF(df['lee_pivots'][~np.isnan(df['lee_pivots'])])(U), 'g', label='Lee', linewidth=3)
                plt.legend()
                plt.plot([0,1], [0,1], 'k--', linewidth=2)
                plt.savefig('lasso_example_variables_exact.pdf')

                plt.clf()
                plt.scatter(df['naive_length'], df['length'])
                plt.savefig('lasso_example_variables_lengths.pdf')

            df.to_csv(csvfile, index=False)
