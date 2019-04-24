import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

# load in the X matrix

from selection.tests.instance import HIV_NRTI
X_full = HIV_NRTI(datafile="NRTI_DATA.txt", standardize=False)[0]

from selection.learning.utils import full_model_inference, liu_inference, pivot_plot
from selection.learning.core import split_sampler, keras_fit
from selection.learning.Rutils import lasso_glmnet, cv_glmnet_lam

boot_design = False

def simulate(s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=6000, seed=0):

    # description of statistical problem

    n, p = X_full.shape

    if boot_design:
        idx = np.random.choice(np.arange(n), n, replace=True)
        X = X_full[idx] # bootstrap X to make it really an IID sample, i.e. don't condition on X throughout
        X += 0.1 * np.std(X) * np.random.standard_normal(X.shape) # to make non-degenerate
    else:
        X = X_full.copy()

    X = X - np.mean(X, 0)[None, :]
    X = X / np.std(X, 0)[None, :]

    n, p = X.shape
    truth = np.zeros(p)
    truth[:s] = np.linspace(signal[0], signal[1], s)
    np.random.shuffle(truth)
    truth /= np.sqrt(n)
    truth *= sigma

    y = X.dot(truth) + sigma * np.random.standard_normal(n)

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    print(dispersion, sigma**2)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(X, XTXi, resid, sampler):

        S = sampler(scale=0.5) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        G = lasso_glmnet(X, ynew, *[None]*4)
        select = G.select(seed=seed)
        return set(list(select[0]))

    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

    # run selection algorithm

    df = full_model_inference(X,
                              y,
                              truth,
                              selection_algorithm,
                              splitting_sampler,
                              success_params=(6, 10),
                              B=B,
                              fit_probability=keras_fit,
                              fit_args={'epochs':10, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'})

    return df

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    init_seed = np.fabs(np.random.standard_normal() * 500)
    for i in range(500):
        df = simulate(seed=init_seed+i)
        csvfile = 'HIV_stability_CV_6000.csv'
        outbase = csvfile[:-4]

        if df is not None or i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass

            if df is not None:
                df.to_csv(csvfile, index=False)

                if len(df['pivot']) > 0:
                    pivot_ax, lengths_ax = pivot_plot(df, outbase)
