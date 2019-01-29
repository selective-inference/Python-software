import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

# load in the X matrix

from selection.tests.instance import HIV_NRTI
X = HIV_NRTI(datafile="NRTI_DATA.txt", standardize=False)[0]
X = X - X.mean(0)[None, :]
X /= np.std(X, 0)[None, :]

from learn_selection.utils import full_model_inference, liu_inference, pivot_plot
from learn_selection.core import split_sampler, keras_fit
from learn_selection.Rutils import lasso_glmnet, cv_glmnet_lam

def simulate(s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=3000, seed=0):

    # description of statistical problem

    n, p = X.shape
    truth = np.zeros(p)
    truth[:s] = np.linspace(signal[0], signal[1], s)
    np.random.shuffle(truth)
    truth /= np.sqrt(n)

    y = sigma * (X.dot(truth) + np.random.standard_normal(n))

    lam_min, lam_1se = cv_glmnet_lam(X, y, seed=seed)
    lam_min, lam_1se = n * lam_min, n * lam_1se

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(X, XTXi, resid, sampler):

        S = sampler(scale=0.) # deterministic with scale=0
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
                              success_params=(1, 1),
                              B=B,
                              fit_probability=keras_fit,
                              fit_args={'epochs':10, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'})

    if df is not None:
        liu_df = liu_inference(X,
                               y,
                               1.00001 * lam_min,
                               dispersion,
                               truth,
                               alpha=alpha)

        return pd.merge(df, liu_df, on='variable')
    
if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    init_seed = np.fabs(np.random.standard_normal() * 500)
    for i in range(500):
        df = simulate(seed=init_seed+i)
        csvfile = 'HIV_CV.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, lengths_ax = pivot_plot(df, outbase)
                liu_pivot = df['liu_pivot']
                liu_pivot = liu_pivot[~np.isnan(liu_pivot)]
                pivot_ax.plot(U, sm.distributions.ECDF(liu_pivot)(U), 'gray', label='Liu CV',
                              linewidth=3)
                pivot_ax.legend()
                fig = pivot_ax.figure
                fig.savefig(csvfile[:-4] + '.pdf')

