import numpy as np
import nose.tools as nt

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim 
from selection.tests.instance import gaussian_instance
import matplotlib.pyplot as plt

def test_onedim_lasso(n=500000, W=1.5, beta=2., sigma=1, randomizer_scale=1):
    """
    Compare to R randomized lasso
    """

    beta = np.array([beta])
    X = np.random.standard_normal((n, 1))
    X /= np.sqrt((X**2).sum(0))[None, :]
    Y = X.dot(beta) + sigma * np.random.standard_normal(n)

    conv = highdim.gaussian(X, 
                            Y, 
                            W * np.ones(X.shape[1]), 
                            randomizer_scale=randomizer_scale * sigma,
                            ridge_term=0.)
    
    signs = conv.fit()
    nonzero = signs != 0

    if nonzero.sum():

        estimate, _, _, pv = conv.selective_MLE(target="full")
        print(estimate, 'selective MLE')
        print(beta[nonzero], 'truth')
        print(np.linalg.pinv(X[:,nonzero]).dot(Y), 'relaxed')
        print(pv)


def main(nsim=500):

    P0, PA = [], []
    from statsmodels.distributions import ECDF

    n, p = 500, 200

    for i in range(nsim):
        try:
            p0, pA = test_highdim_lasso(n=n, p=p, full=True)
        except:
            p0, pA = [], []
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(P0), np.std(P0), np.mean(np.array(PA) < 0.05))
    
        if i % 3 == 0 and i > 0:
            U = np.linspace(0, 1, 101)
            plt.clf()
            if len(P0) > 0:
                plt.plot(U, ECDF(P0)(U))
            if len(PA) > 0:
                plt.plot(U, ECDF(PA)(U), 'r')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.savefig("plot.pdf")
    plt.show()

