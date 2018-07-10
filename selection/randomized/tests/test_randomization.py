from __future__ import print_function

import numpy as np
import nose.tools as nt

from selection.randomized.randomization import randomization

def test_noise_dbns():

    X = np.random.standard_normal((10, 5))
    Q = X.T.dot(X)
    noises = [randomization.isotropic_gaussian((5,), 1.),
              randomization.laplace((5,), 1.),
              randomization.logistic((5,), 1.),
              randomization.gaussian(Q)]

    v1, v2 = [], []

    for i, noise in enumerate(noises):

        x = np.random.standard_normal(5)
        u = np.random.standard_normal(5)
        v1.append(np.exp(noise.log_density(x)))
        v2.append(noise._density(x))

        noise.smooth_objective(x, 'func')
        noise.smooth_objective(x, 'grad')
        noise.smooth_objective(x, 'both')
        noise.gradient(x)

        nt.assert_equal(noise.sample().shape, (5,))
        nt.assert_equal(noise.sample().shape, (5,))

        if noise.CGF is not None:
            u = np.zeros(5)
            u[:2] = 0.1
            noise.CGF.smooth_objective(u, 'both')

        if noise.CGF_conjugate is not None:
            noise.CGF_conjugate.smooth_objective(x, 'both')


