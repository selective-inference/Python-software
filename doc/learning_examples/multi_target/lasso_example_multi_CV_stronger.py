import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.utils import full_model_inference, pivot_plot
from selection.learning.core import split_sampler, keras_fit
from selection.learning.Rutils import cv_glmnet_lam, lasso_glmnet

def simulate(n=200, p=100, s=10, signal=(1.5, 2), sigma=2, alpha=0.1, B=3000):

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

    return full_model_inference(X,
                                y,
                                truth,
                                selection_algorithm,
                                splitting_sampler,
                                success_params=(1, 1),
                                B=B,
                                fit_probability=keras_fit,
                                fit_args={'epochs':10, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'})

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    for i in range(500):
        df = simulate()
        csvfile = 'lasso_multi_CV_stronger.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_plot(df, outbase)
