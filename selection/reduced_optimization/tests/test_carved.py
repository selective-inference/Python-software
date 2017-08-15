import numpy as np
import regreg.api as rr

from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.decorators import (set_seed_iftrue, 
                                 set_sampling_params_iftrue)

from ..estimator import M_estimator_approx_carved
from ...tests.instance import logistic_instance, gaussian_instance

@set_seed_iftrue(SET_SEED)
def test_carved():
    n = 500
    p = 100
    s = 0
    signal = 0.

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0, signal=signal)
    lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

    n, p = X.shape

    loss = rr.glm.gaussian(X, y)
    total_size = loss.saturated_loss.shape[0]
    subsample_size = int(0.8* total_size)
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
    M_est = M_estimator_approx_carved(loss, epsilon, subsample_size, penalty, 'parametric')
    M_est.solve_approx()
