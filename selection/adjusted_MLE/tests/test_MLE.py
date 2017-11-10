from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, selective_MLE
import matplotlib.pyplot as plt

def test(n=100, p=1, s=1, signal=5., seed_n = 0, lam_frac=1., randomization_scale=1.):
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
    n, p = X.shape
    np.random.seed(seed_n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
    #randomizer = randomization.gaussian(np.identity(p))
    M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale)

    M_est.solve_approx()
    active = M_est._overall
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    if nactive>0:
        solve_mle = selective_MLE(M_est)
        mle = np.zeros(nactive)
        for j in range(nactive):
            mle[j] = solve_mle.solve_UMVU(j)[0]

        return mle, M_est.target_observed
    else:
        return None

print(test())
def simulate(ndraw = 100):
    seed_seq = np.arange(ndraw)
    sel_MLE = []
    naive_MLE = []
    for i in range(seed_seq.shape[0]):
        draw = test(n=100, p=1, s=1, signal=0., seed_n = seed_seq[i])
        if draw[0] is not None:
            sel_MLE.append(draw[0])
            naive_MLE.append(draw[1])

    plt.plot(np.asarray(naive_MLE), np.asarray(sel_MLE), 'r--')
    plt.show()

#simulate()