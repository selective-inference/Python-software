import functools, uuid

import numpy as np, pandas as pd
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance


from selection.learning.utils import full_model_inference, pivot_plot
from selection.learning.core import split_sampler, keras_fit

from sklearn.linear_model import lasso_path

def simulate(n=2000, p=1000, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=2000):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p,
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.1,
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=True)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTX, XTXi, sampler):

        min_success = 6
        ntries = 10

        def _alpha_grid(X, y, center, XTX):
            n, p = X.shape
            alphas, coefs, _ = lasso_path(X, y, Xy=center, precompute=XTX)
            nselected = np.count_nonzero(coefs, axis=0)
            return alphas[nselected < np.sqrt(0.8 * p)]

        alpha_grid = _alpha_grid(X, y, sampler(scale=0.), XTX)
        success = np.zeros((p, alpha_grid.shape[0]))

        for _ in range(ntries):
            scale = 1.  # corresponds to sub-samples of 50%
            noisy_S = sampler(scale=scale)
            _, coefs, _ = lasso_path(X, y, Xy = noisy_S, precompute=XTX, alphas=alpha_grid)
            success += np.abs(np.sign(coefs))

        selected = np.apply_along_axis(lambda row: any(x>min_success for x in row), 1, success)
        vars = set(np.nonzero(selected)[0])
        return vars

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)

    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi)

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

    for i in range(500):
        df = simulate(B=3000)
        csvfile = 'stability_selection_harder_big.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)


