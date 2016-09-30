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

    for noise in noises:
        x = np.random.standard_normal(5)
        u = np.random.standard_normal(5)
        noise.smooth_objective(x, 'func')
        noise.smooth_objective(x, 'grad')
        noise.smooth_objective(x, 'both')
        noise.gradient(x)

        nt.assert_equal(noise.sample().shape, (5,))

        if hasattr(noise, "CGF"):
            val, grad = noise.CGF
            u = np.zeros(5)
            u[:2] = 0.1
            val(u), grad(u)

        if hasattr(noise, "CGF_grad"):
            val, grad = noise.CGF_grad
            val(x), grad(x)
