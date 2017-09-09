from __future__ import print_function
import numpy as np

from selection.randomized.api import lasso as randomized_lasso
from selection.tests.instance import gaussian_instance

def test_randomized_lasso(n=300, p=500, s=5, signal=7.5, rho=0.2):

    X, Y, beta, active, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, equicorrelated=False)

    L = randomized_lasso.gaussian(X, Y, 3.5 * sigma * np.ones(p))
    signs = L.fit()

    print(np.nonzero(signs != 0)[0])
    print(np.nonzero(beta != 0)[0])
    print(L.summary(signs != 0, ndraw=1000, burnin=200, compute_intervals=False))


if __name__ == "__main__":
    test_randomized_lasso()
