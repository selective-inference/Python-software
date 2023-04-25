import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selectinf.tests.instance import gaussian_instance

from selectinf.learning.utils import full_model_inference, pivot_plot
from selectinf.learning.core import normal_sampler
from selectinf.learning.Rfitters import logit_fit
from selectinf.learning.learners import mixture_learner
mixture_learner.scales = [1]*10 + [1.5,2,3,4,5,10]

def BHfilter(pval, q=0.2):
    pval = np.asarray(pval)
    pval_sort = np.sort(pval)
    comparison = q * np.arange(1, pval.shape[0] + 1.) / pval.shape[0]
    passing = pval_sort < comparison
    if passing.sum():
        thresh = comparison[np.nonzero(passing)[0].max()]
        return np.nonzero(pval <= thresh)[0]
    return []

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

def simulate(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=1000):

    # description of statistical problem

    X, y, truth = generate(n=n, p=p, s=s, signal=signal, sigma=sigma)

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)

    def meta_algorithm(XTX, XTXi, dispersion, lam, sampler):
        global counter
        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.
        noisy_S = sampler(scale=scale)
        soln = XTXi.dot(noisy_S)
        solnZ = soln / (np.sqrt(np.diag(XTXi)) * np.sqrt(dispersion))
        pval = ndist.cdf(solnZ)
        pval = 2 * np.minimum(pval, 1 - pval)
        return set(BHfilter(pval, q=0.2))

    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, dispersion, lam)

    # run selection algorithm

    return full_model_inference(X,
                                y,
                                truth,
                                selection_algorithm,
                                smooth_sampler,
                                success_params=(1, 1),
                                B=B,
                                fit_probability=logit_fit,
                                fit_args={'df':20},
                                how_many=1)

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    opts = dict(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=5000)

    R2 = []
    for _ in range(100):

        X, y, truth = generate(**opts)
        R2.append((np.linalg.norm(y-X.dot(truth))**2, np.linalg.norm(y)**2))

    R2 = np.array(R2)
    R2mean = 1 - np.mean(R2[:,0]) / np.mean(R2[:,1])
    print('R2', R2mean)

    for i in range(5000):
        df = simulate(**opts)
        csvfile = __file__[:-3] + '.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)
            df['R2'] = np.ones(df.shape[0]) * R2mean
            if len(df['pivot']) > 0:
                f = pivot_plot(df, outbase)[1]
                plt.close(f)

