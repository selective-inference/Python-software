import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.utils import (full_model_inference, 
                                   pivot_plot,
                                   liu_inference)

from selection.learning.core import split_sampler, logit_fit

def simulate(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1):

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

    def meta_algorithm(XTX, XTXi, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.
        noisy_S = sampler(scale=scale)
        loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve(max_its=50, tol=1.e-6)
        success += soln != 0
        return set(np.nonzero(success)[0])

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, lam)

                                fit_probability=keras_fit,
                                fit_args={'epochs':200},

    # run selection algorithm


    df = full_model_inference(X,
                              y,
                              truth,
                              selection_algorithm,
                              splitting_sampler,
                              success_params=(1, 1),
                              B=B,
                              fit_probability=logit_fit,
                              fit_args={'df':20})

    if df is not None:

        liu_df = liu_inference(X,
                               y,
                               lam,
                               dispersion,
                               truth,
                               alpha=alpha)

        return pd.merge(df, liu_df, on='variable')

liu_df = 

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(2000):
        df = simulate(B=2000)
        csvfile = 'keras.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)


