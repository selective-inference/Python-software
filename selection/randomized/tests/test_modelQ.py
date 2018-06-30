from __future__ import division, print_function

import numpy as np
import nose.tools as nt

import regreg.api as rr

from ..modelQ import modelQ
from ..lasso import lasso
from ...tests.instance import gaussian_instance

def test_modelQ():

    n, p, s = 200, 50, 4
    X, y, beta = gaussian_instance(n=n,
                                   p=p,
                                   s=s,
                                   sigma=1)[:3]

    lagrange = 5. * np.ones(p) * np.sqrt(n)
    perturb = np.random.standard_normal(p) * n
    LH = lasso.gaussian(X, y, lagrange)
    LH.fit(perturb=perturb, solve_args={'min_its':1000})

    LQ = modelQ(X.T.dot(X), X, y, lagrange)
    LQ.fit(perturb=perturb, solve_args={'min_its':1000})
    LQ.summary() # smoke test

    conH = LH.sampler.affine_con
    conQ = LQ.sampler.affine_con

    np.testing.assert_allclose(LH.initial_soln, LQ.initial_soln)
    np.testing.assert_allclose(LH.initial_subgrad, LQ.initial_subgrad)

    np.testing.assert_allclose(conH.linear_part, conQ.linear_part)
    np.testing.assert_allclose(conH.offset, conQ.offset)

    np.testing.assert_allclose(LH._beta_full, LQ._beta_full)

