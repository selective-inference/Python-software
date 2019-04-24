import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.utils import full_model_inference, pivot_plot
from selection.learning.core import normal_sampler, keras_fit

def simulate(n=1000, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, seed=0, B=5000):

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

        rho = 0.8
        S = sampler(scale=0.) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        Xnew = rho * X + np.sqrt(1 - rho**2) * np.random.standard_normal(X.shape)

        X_full = np.hstack([X, Xnew])
        beta_full = np.linalg.pinv(X_full).dot(ynew)
        winners = np.fabs(beta_full)[:p] > np.fabs(beta_full)[p:]
        return set(np.nonzero(winners)[0])

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)


    # run selection algorithm

    return full_model_inference(X,
                                y,
                                truth,
                                selection_algorithm,
                                smooth_sampler,
                                success_params=(8, 10),
                                B=B,
                                fit_probability=keras_fit,
                                fit_args={'epochs':20, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'})

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    iseed = int(np.fabs(np.random.standard_normal() * 50000))
    for i in range(500):
        df = simulate(seed=i + iseed, B=3000)
        csvfile = 'knockoff_kernel_multi.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)


