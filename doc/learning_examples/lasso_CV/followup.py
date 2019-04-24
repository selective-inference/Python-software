import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.Rutils import lasso_glmnet
from selection.learning.utils import (full_model_inference, 
                                   pivot_plot,
                                   naive_full_model_inference)
from selection.learning.core import normal_sampler, probit_fit

def simulate(n=400, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, seed=0, B=2000):

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

    df = full_model_inference(X,
                              y,
                              truth,
                              selection_algorithm,
                              smooth_sampler,
                              success_params=(1, 1),
                              B=B,
                              fit_probability=probit_fit,
                              fit_args={'df':20},
                              how_many=1,
                              naive=False)

    if df is not None:

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

        naive_df = naive_full_model_inference(X2,
                                              y2,
                                              dispersion_2,
                                              truth,
                                              df['variable'],
                                              alpha=alpha,
                                              how_many=1)

        return pd.merge(df, naive_df, on='variable')

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    iseed = int(np.fabs(np.random.standard_normal() * 1000))
    for i in range(500):
        df = simulate(seed=i + iseed)
        csvfile = 'followup.csv'

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

