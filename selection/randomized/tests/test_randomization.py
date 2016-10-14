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

    for i, noise in enumerate(noises):

        x = np.random.standard_normal(5)
        u = np.random.standard_normal(5)
        noise.log_density(x)
        noise.smooth_objective(x, 'func')
        noise.smooth_objective(x, 'grad')
        noise.smooth_objective(x, 'both')
        noise.gradient(x)

        S = noise.sample()
        if i != 3:
            np.testing.assert_allclose(float(noise.log_density(S)), float(np.log(noise._density(S)).sum()))
        nt.assert_equal(noise.sample().shape, (5,))
        nt.assert_equal(noise.sample().shape, (5,))

        if hasattr(noise, "CGF"):
            val, grad = noise.CGF
            u = np.zeros(5)
            u[:2] = 0.1
            val(u), grad(u)

        if hasattr(noise, "CGF_grad"):
            val, grad = noise.CGF_grad
            val(x), grad(x)
