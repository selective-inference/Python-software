from __future__ import print_function

import numpy as np
import sys
import regreg.api as rr

import selection.tests.reports as reports
from ...randomized.api import randomization
from ...tests.instance import logistic_instance, gaussian_instance
from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue

from ..ci_approx_density import approximate_conditional_density
from ..approx_ci.estimator_approx import M_estimator_approx

from ...randomized.query import naive_confidence_intervals
from ...randomized.query import naive_pvalues


@register_report(['cover', 'ci_length', 'truth', 'naive_cover', 'naive_pvalues'])
@wait_for_return_value()
def test_approximate_ci(n=100,
                        p=10,
                        s=3,
                        snr=5,
                        rho=0.1,
                        lam_frac = 1.,
                        loss='gaussian',
                        randomizer='gaussian'):


    if loss == "gaussian":
        X, y, beta = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr)[:3]
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)[:]
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
        loss = rr.glm.logistic(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=randomization_scale)
    M_est = M_estimator_map(loss, epsilon, penalty, randomization, randomization_scale = randomization_scale)

    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
    sys.stderr.write("Observed target" + str(M_est.target_observed) + "\n")

    if nactive == 0:
        return None

    else:
        true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(true_mean)

        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_covered = np.zeros(nactive)
        naive_risk = np.zeros(nactive)

        ci = approximate_conditional_density(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        sel_MLE = np.zeros(nactive)
        sel_length = np.zeros(nactive)

        for j in range(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            sel_MLE[j] = ci.approx_MLE_solver(j, step=1, nstep=150)[0]
            sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]

        sel_covered = np.zeros(nactive, np.bool)
        sel_risk = np.zeros(nactive)

        for j in range(nactive):
            sel_risk[j] = (sel_MLE[j] - true_vec[j]) ** 2.
            naive_risk[j] = (M_est.target_observed[j]- true_vec[j]) ** 2.

            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                sel_covered[j] = 1
            if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                naive_covered[j] = 1

        print("lengths", sel_length.sum()/nactive)
        print("selective intervals", ci_sel.T)
        print("risks", sel_risk.sum()/nactive)

        return np.transpose(np.vstack((ci_sel[:, 0],
                                       ci_sel[:, 1],
                                       ci_naive[:,0],
                                       ci_naive[:, 1],
                                       sel_MLE,
                                       M_est.target_observed,
                                       sel_covered,
                                       naive_covered,
                                       sel_risk,
                                       naive_risk)))


def test_lasso(n, p, s, signal):
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
    true_mean = X.dot(beta)
    lasso = test_approximate_inference(X,
                                       y,
                                       true_mean,
                                       sigma,
                                       seed_n=0,
                                       lam_frac=1.,
                                       loss='gaussian')

    if lasso is not None:
        print("output of selection adjusted inference", lasso)
        return(lasso)


