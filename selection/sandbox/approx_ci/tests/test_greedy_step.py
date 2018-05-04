from __future__ import print_function
import sys
import numpy as np
import regreg.api as rr
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.selection_map import greedy_score_map
from selection.approx_ci.ci_approx_greedy_step import approximate_conditional_density


from selection.randomized.query import naive_confidence_intervals

def approximate_inference(X,
                          y,
                          beta,
                          sigma,
                          seed_n = 0,
                          lam_frac = 1.,
                          loss='gaussian',
                          randomization_scale = 1.):

    from selection.api import randomization
    n, p = X.shape
    np.random.seed(seed_n)
    if loss == "gaussian":
        loss = rr.glm.gaussian(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    elif loss == "logistic":
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    randomization = randomization.isotropic_gaussian((p,), scale=randomization_scale)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    GS = greedy_score_map(loss,
                          penalty,
                          np.zeros(p, dtype=bool),
                          np.ones(p, dtype=bool),
                          randomization,
                          randomization_scale)

    GS.solve_approx()
    active = GS._overall
    nactive = np.sum(active)

    if nactive == 0:
        return None
    else:
        active_set = np.asarray([i for i in range(p) if active[i]])
        s = beta.sum()
        true_support = np.asarray([i for i in range(p) if i < s])
        true_vec = beta[active]

        if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:
            ci = approximate_conditional_density(GS)
            ci.solve_approx()
            sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

            ci_naive = naive_confidence_intervals(GS.target_cov, GS.target_observed)
            naive_covered = np.zeros(nactive)
            naive_risk = np.zeros(nactive)

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
                naive_risk[j] = (GS.target_observed[j] - true_vec[j]) ** 2.

                if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                    sel_covered[j] = 1
                if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                    naive_covered[j] = 1

            print("lengths", sel_length.sum() / nactive)
            print("selective intervals", ci_sel.T)
            print("risks", sel_risk.sum() / nactive)

            return np.transpose(np.vstack((ci_sel[:, 0],
                                           ci_sel[:, 1],
                                           ci_naive[:, 0],
                                           ci_naive[:, 1],
                                           sel_MLE,
                                           GS.target_observed,
                                           sel_covered,
                                           naive_covered,
                                           sel_risk,
                                           naive_risk)))


def test_greedy_step(n=50, p=100, s=5, signal=5):
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0., signal=signal, sigma=1.)
    greedy_step = approximate_inference(X,
                                        y,
                                        beta,
                                        sigma,
                                        seed_n=0,
                                        lam_frac=1.,
                                        loss='gaussian')

    if greedy_step is not None:
        print("output of selection adjusted inference", greedy_step)
        return(greedy_step)
