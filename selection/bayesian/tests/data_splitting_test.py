from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.randomized.M_estimator import M_estimator
from selection.bayesian.inference_rr_data_split import smooth_cube_barrier, selection_probability_split
from selection.randomized.randomization import split


def test_sel_prob_split(n=100, p=20, s=5, snr=5, rho=0.1,lam_frac=1.,loss='gaussian'):

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
    lagrange = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lagrange
    penalty = rr.group_lasso(np.arange(p),weights=dict(zip(np.arange(p), W)), lagrange=1.)

    total_size = loss.saturated_loss.shape[0]

    subsample_size = int(0.8* total_size)

    randomizer_split = split(loss.shape, subsample_size, total_size)

    solver = M_estimator(loss, epsilon, penalty, randomizer_split)

    solver.Msolve()

    active_set = np.asarray([i for i in range(p) if solver._overall[i]])

    true_support = np.asarray([i for i in range(p) if i < s])

    print("active set, true_support", active_set, true_support)

    if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:

        generative_mean = np.append(snr*np.ones(s), np.zeros(p-s))
        sel_split = selection_probability_split(loss, epsilon, penalty, generative_mean)
        sel_prob_split = sel_split.minimize2(nstep=100)[::-1]
        print("sel prob and minimizer", sel_prob_split[0], sel_prob_split[1])

print(test_sel_prob_split())