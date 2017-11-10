from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
from selection.adjusted_MLE.tests.exact_MLE import grad_CGF

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

    M_est.solve_map()
    active = M_est._overall
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    if nactive > 0:
        mle = solve_UMVU(M_est.target_transform,
                         M_est.opt_transform,
                         M_est.target_observed,
                         M_est.feasible_point,
                         M_est.target_cov,
                         M_est.randomizer_precision)

        return mle[0], M_est.target_observed, nactive
    else:
        return None

def test_selective_MLE(target_observed=2):

    """
    Simple problem thresholded at 2
    """

    target_transform = (np.identity(1), np.zeros(1))
    opt_transform = (np.identity(1), -np.ones(1) * 2.)
    feasible_point = 1.
    randomizer_precision = np.identity(1)
    target_cov = np.identity(1)

    return solve_UMVU(target_transform,
                      opt_transform,
                      target_observed,
                      feasible_point,
                      target_cov,
                      randomizer_precision)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    Zval = np.linspace(-1,5,51)

    mu_seq = np.linspace(-7., 6, num=2600)
    grad_partition = np.array([grad_CGF(mu, randomization_scale = 1., threshold = 2) for mu in mu_seq])

    exact_MLE = []
    for k in range(Zval.shape[0]):
        true = mu_seq[np.argmin(np.abs(grad_partition - Zval[k]))]
        exact_MLE.append(true)

    MLE = np.array([test_selective_MLE(z)[0] for z in Zval])
    MLE = MLE * (np.fabs(MLE) < 200)

    plt.plot(Zval, MLE)
    plt.plot(Zval, np.asarray(exact_MLE), 'r--')
    plt.show()
