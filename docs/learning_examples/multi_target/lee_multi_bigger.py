import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.utils import (partial_model_inference, 
                                   pivot_plot,
                                   lee_inference)
from selection.learning.core import normal_sampler, keras_fit
from selection.learning.learners import sparse_mixture_learner

def simulate(n=2000, p=300, s=10, signal=(3 / np.sqrt(2000), 4 / np.sqrt(2000)), sigma=2, alpha=0.1, B=10000):

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
    print(np.linalg.norm(truth))

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)

    def meta_algorithm(XTX, XTXi, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.
        noisy_S = sampler(scale=scale)
        loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve(max_its=300, tol=1.e-10)
        success += soln != 0
        return tuple(sorted(np.nonzero(success)[0]))

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, lam)

    # run selection algorithm

    df = partial_model_inference(X, 
                                 y,
                                 truth,
                                 selection_algorithm,
                                 smooth_sampler,
                                 fit_probability=keras_fit,
                                 fit_args={'epochs':30, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'},
                                 success_params=(1, 1),
                                 B=B,
                                 alpha=alpha,
                                 learner_klass=sparse_mixture_learner)

    lee_df = lee_inference(X,
                           y,
                           lam,
                           dispersion,
                           truth,
                           alpha=alpha)

    return pd.merge(df, lee_df, on='variable')


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    for i in range(500):
        df = simulate()
        csvfile = 'lee_multi_bigger.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)
                #pivot_ax.plot(U, sm.distributions.ECDF(df['lee_pivot'][~np.isnan(df['lee_pivot'])])(U), 'g', label='Lee', linewidth=3)
                pivot_ax.figure.savefig(outbase + '.pdf')

                length_ax.scatter(df['naive_length'], df['lee_length'])
                length_ax.figure.savefig(outbase + '_lengths.pdf')
