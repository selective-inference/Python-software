import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selectinf.tests.instance import gaussian_instance

from selectinf.learning.Rutils import lasso_glmnet
from selectinf.learning.utils import (full_model_inference, 
                                      pivot_plot,
                                      split_full_model_inference)
from selectinf.learning.core import normal_sampler, keras_fit
from selectinf.learning.fitters import gbm_fit_sk

def generate(n=2000, p=100, s=10, signal=(np.sqrt(2)*0.5, np.sqrt(2)*1), sigma=2, **ignored):

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False)[:3]

    return X, y, truth

def simulate(n=2000, p=100, s=10, signal=(np.sqrt(2)*0.5, np.sqrt(2)*1), 
             sigma=2, alpha=0.1,B=3000):

    # description of statistical problem

    X, y, truth = generate(n=n,
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


    def meta_algorithm(X, XTXi, resid, sampler):

        n, p = X.shape
        idx = np.random.choice(np.arange(n), int(n/2), replace=False)

        S = sampler(scale=0.) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        Xidx, yidx = X[idx], y[idx]
        rho = 0.8

        Xnew = rho * Xidx + np.sqrt(1 - rho**2) * np.random.standard_normal(Xidx.shape)

        X_full = np.hstack([Xidx, Xnew])
        beta_full = np.linalg.pinv(X_full).dot(yidx)
        winners = np.fabs(beta_full)[:p] > np.fabs(beta_full)[p:]
        return set(np.nonzero(winners)[0])

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
                              success_params=(8, 10),
                              B=B,
                              fit_probability=gbm_fit_sk,
                              fit_args={'n_estimators':1000}
                              )

    if df is not None:

        observed_set = list(df['variable'])
        idx2 = np.random.choice(np.arange(n), int(n/2), replace=False)
        split_df = split_full_model_inference(X,
                                              y,
                                              idx2,
                                              None, # ignored dispersion
                                              truth,
                                              observed_set,
                                              alpha=alpha)

        df = pd.merge(df, split_df, on='variable')
        return df

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    opts = dict(n=2000, p=100, s=10, 
                signal=(np.sqrt(2)*0.5, np.sqrt(2)*1), sigma=2, 
                alpha=0.1, B=6000)

    R2 = []
    for _ in range(100):

        X, y, truth = generate(**opts)
        R2.append((np.linalg.norm(y-X.dot(truth))**2, np.linalg.norm(y)**2))

    R2 = np.array(R2)
    R2mean = 1 - np.mean(R2[:,0]) / np.mean(R2[:,1])
    print('R2', R2mean)


    for i in range(5000):
        df = simulate(**opts)
        csvfile = __file__[:-3] + '_gbm.csv'
        outbase = csvfile[:-4]

        if df is not None:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                f = pivot_plot(df, outbase)[1]
                plt.close(f)

