import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selectinf.tests.instance import gaussian_instance

from selectinf.learning.utils import (full_model_inference, 
                                      pivot_plot, 
                                      split_full_model_inference)
from selectinf.learning.core import normal_sampler, keras_fit
from selectinf.learning.Rutils import lasso_glmnet

def simulate(n=1000, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=2000):

    # description of statistical problem

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

    idx = np.random.choice(np.arange(n), int(n/2), replace=False)

    def meta_algorithm(X, XTXi, resid, idx, sampler):

        n, p = X.shape

        S = sampler(scale=0.) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X

        G = lasso_glmnet(X[idx], ynew[idx], *[None]*4)
        select = G.select()
        return set(list(select[0]))

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid, idx)

    # run selection algorithm

    df = full_model_inference(X,
                              y,
                              truth,
                              selection_algorithm,
                              smooth_sampler,
                              success_params=(1, 1),
                              B=B,
                              fit_probability=keras_fit,
                              fit_args={'epochs':20, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'})
    

    if df is not None:

        observed_set = list(df['variable'])
        split_df = split_full_model_inference(X, 
                                              y,
                                              idx,
                                              dispersion,
                                              truth,
                                              observed_set,
                                              alpha=alpha)

        df = pd.merge(df, split_df, on='variable')
        return df

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(500):
        df = simulate(B=3000)
        csvfile = __file__[:-3] + '.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                f = pivot_plot(df, outbase)[1]
                plt.close(f)


