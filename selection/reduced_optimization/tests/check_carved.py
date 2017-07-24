from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
from selection.reduced_optimization.estimator import M_estimator_approx_carved
from selection.tests.instance import logistic_instance, gaussian_instance


n = 500
p = 100
s = 0
snr = 0.

X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)
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
