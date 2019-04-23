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

def simulate(s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=5000, seed=0):

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
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTX, XTXi, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.
        noisy_S = sampler(scale=scale)
        loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve(max_its=100, tol=1.e-10)
        success += soln != 0
        return set(np.nonzero(success)[0])

    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, lam)

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

    if False: # df is not None:
        liu_df = liu_inference(X,
                               y,
                               lam,
                               dispersion,
                               truth,
                               alpha=alpha)

        return pd.merge(df, liu_df, on='variable')
    else:
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
        csvfile = 'HIV_fixed.csv'
        outbase = csvfile[:-4]

        if df is not None or i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, lengths_ax = pivot_plot(df, outbase)
#                 liu_pivot = df['liu_pivot']
#                 liu_pivot = liu_pivot[~np.isnan(liu_pivot)]
#                 pivot_ax.plot(U, sm.distributions.ECDF(liu_pivot)(U), 'gray', label='Liu CV',
#                               linewidth=3)
#                 pivot_ax.legend()
#                 fig = pivot_ax.figure
#                 fig.savefig(csvfile[:-4] + '.pdf')

