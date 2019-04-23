import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.utils import full_model_inference, pivot_plot
from selection.learning.core import split_sampler, keras_fit
from selection.learning.learners import mixture_learner
mixture_learner.scales = [1]*10 + [1.5,2,3,4,5,10]

def simulate(n=200, p=50, s=5, signal=(0.5, 1), sigma=2, alpha=0.1, B=1000):

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

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTX, XTXi, dispersion, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.5
        noisy_S = sampler(scale=scale)
        soln = XTXi.dot(noisy_S)
        solnZ = soln / (np.sqrt(np.diag(XTXi)) * np.sqrt(dispersion))
        return set(np.nonzero(np.fabs(solnZ) > 2.1)[0])

    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, dispersion, lam)

    # run selection algorithm

    return full_model_inference(X,
                                y,
                                truth,
                                selection_algorithm,
                                splitting_sampler,
                                success_params=(5, 7),
                                B=B,
                                fit_probability=keras_fit,
                                fit_args={'epochs':30, 'sizes':[100, 100], 'activation':'relu'})


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(500):
        df = simulate(B=10000)
        csvfile = 'keras_targets_medium.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)
