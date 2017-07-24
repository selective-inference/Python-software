from __future__ import print_function
import numpy as np

from selection.randomized.api import lasso as randomized_lasso
from selection.tests.instance import gaussian_instance

def test_randomized_lasso(n=100, p=200, s=10, signal=7, rho=0):

    X, Y, beta, active, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho)

    L = randomized_lasso.gaussian(X, Y, 4.5 * sigma * np.ones(p))
    signs = L.fit()

    print(L.summary(signs != 0))


if __name__ == "__main__":
    test_randomized_lasso()
