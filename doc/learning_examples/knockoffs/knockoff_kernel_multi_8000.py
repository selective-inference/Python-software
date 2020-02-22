import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selectinf.tests.instance import gaussian_instance

from selectinf.learning.utils import full_model_inference, pivot_plot
from selectinf.learning.core import normal_sampler, keras_fit

def generate(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, **ignored):

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

def simulate(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=3000):

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

    opts = dict(n=2000, p=100, s=10, signal=(0.5, 1), 
                sigma=2, alpha=0.1, B=8000)

    R2 = []
    for _ in range(100):

        X, y, truth = generate(**opts)
        R2.append((np.linalg.norm(y-X.dot(truth))**2, np.linalg.norm(y)**2))

    R2 = np.array(R2)
    R2mean = 1 - np.mean(R2[:,0]) / np.mean(R2[:,1])
    print('R2', R2mean)


    iseed = int(np.fabs(np.random.standard_normal() * 50000))
    for i in range(2000):
        df = simulate(**opts)
        csvfile = __file__[:-3] + '_2000.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                f = pivot_plot(df, outbase)[1]
                plt.close(f)

