import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selectinf.tests.instance import gaussian_instance

from selectinf.learning.utils import full_model_inference, pivot_plot
from selectinf.learning.core import split_sampler, keras_fit
from selectinf.learning.Rutils import lasso_glmnet
from rpy2.robjects import numpy2ri
import rpy2.robjects as rpy

class lasso_glmnet_split(lasso_glmnet):

    def select(self, CV=True, seed=0):

        numpy2ri.activate()

        rpy.r.assign('X', self.X.copy())
        rpy.r.assign('Y', self.Y.copy())
        rpy.r('X = as.matrix(X)')
        rpy.r('Y = as.numeric(Y)')
        rpy.r('n = nrow(X)')
        rpy.r('split_ = sample(1:n, n/2, replace=FALSE)')
        rpy.r('Xsplit_ = X[split_,]')
        rpy.r('Ysplit_ = Y[split_]')
        rpy.r('set.seed(%d)' % seed)
        rpy.r('cvG = cv.glmnet(Xsplit_, Ysplit_, intercept=FALSE, standardize=FALSE)')
        rpy.r("L1 = cvG[['lambda.min']]")
        rpy.r("L2 = cvG[['lambda.1se']]")
        if CV:
            rpy.r("L = L1")
        else:
            rpy.r("L = 0.99 * L2")
        rpy.r("G = glmnet(X, Y, intercept=FALSE, standardize=FALSE)")
        n, p = self.X.shape
        L = rpy.r('L')
        rpy.r('B = as.numeric(coef(G, s=L, exact=TRUE, x=X, y=Y))[-1]')
        B = np.asarray(rpy.r('B'))
        selected = (B != 0)
        numpy2ri.deactivate()
        if selected.sum():
            V = np.nonzero(selected)[0]
            return V, V
        else:
            return [], []


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
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(X, XTXi, resid, sampler):

        S = sampler(scale=0.) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        G = lasso_glmnet_split(X, ynew, *[None]*4)
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

    opts = dict(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=2000)

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

        if df is not None:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                f = pivot_plot(df, outbase)[1]
                plt.close(f)

