import numpy as np
import nose.tools as nt
import numpy.testing.decorators as dec

from selection.tests.instance import gaussian_instance as instance
import selection.tests.reports as reports

from selection.algorithms.lasso import lasso 
from selection.algorithms.debiased_lasso import debiased_lasso_inference
import regreg.api as rr

def test_gaussian(n=100, p=20):

    X, y, beta = instance(n=n, p=p, sigma=1.)[:3]

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))

    weights = 1.1 * lam_theor * np.ones(p)
    weights[:3] = 0.

    L = lasso.gaussian(X, y, weights, sigma=1.)
    L.ignore_inactive_constraints = True
    L.fit()

    print(debiased_lasso_inference(L, L.active, np.sqrt(2 * np.log(p) / n)))
    print(beta)
