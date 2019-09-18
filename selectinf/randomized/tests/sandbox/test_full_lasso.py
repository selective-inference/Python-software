import numpy as np
import nose.tools as nt

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import lasso
from selection.tests.instance import gaussian_instance
import matplotlib.pyplot as plt

def test_full_lasso(n=200, p=30, signal_fac=1.5, s=5, ndraw=5000, burnin=1000, sigma=3, full=False, rho=0.4, randomizer_scale=1):
    """
    General LASSO -- 
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    n, p = X.shape

    W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer_scale=randomizer_scale * sigma)
    
    signs = conv.fit(solve_args={'min_its':500, 'tol':1.e-13})
    nonzero = signs != 0

    conv2 = lasso.gaussian(X, 
                           Y, 
                           W,
                           randomizer_scale=randomizer_scale * sigma)
    conv2.fit(perturb=conv._initial_omega, solve_args={'min_its':500, 'tol':1.e-13})
    conv2.decompose_subgradient(condition=np.ones(p, np.bool))

    np.testing.assert_allclose(conv2._view.sampler.affine_con.covariance,
                               conv.sampler.affine_con.covariance)

    np.testing.assert_allclose(conv2._view.sampler.affine_con.mean,
                               conv.sampler.affine_con.mean)

    np.testing.assert_allclose(conv2._view.sampler.affine_con.linear_part,
                               conv.sampler.affine_con.linear_part)

    np.testing.assert_allclose(conv2._view.sampler.affine_con.offset,
                               conv.sampler.affine_con.offset)

    np.testing.assert_allclose(conv2._view.initial_soln,
                               conv.initial_soln)

    np.testing.assert_allclose(conv2._view.initial_subgrad,
                               conv.initial_subgrad)
