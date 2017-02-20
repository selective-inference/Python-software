from __future__ import print_function
import time

import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.bayesian.cisEQTLS.inference_2sels import selection_probability_genes_variants, \
    sel_prob_gradient_map_simes_lasso, selective_inf_simes_lasso
from scipy.stats import norm as normal

def sel_prob_ms_lasso():
    n = 100
    p = 100
    s = 0
    snr = 0.

    X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

    n, p = X.shape

    alpha = 0.05

    sel_simes = simes_selection(X, y, alpha=0.05, randomizer='gaussian')

    if sel_simes is not None:

        index = sel_simes[0]

        t_0 = sel_simes[2]

        J = sel_simes[1]

        T_sign = sel_simes[3]*np.ones(1)

        T_stats = sel_simes[4]*np.ones(1)

        if t_0 == 0:
            threshold = normal.ppf(1.- alpha/(2.*p))*np.ones(1)

        else:
            J_card = J.shape[0]
            threshold = np.zeros(J_card+1)
            threshold[:J_card] = normal.ppf(1. - (alpha/(2.*p))*(np.arange(J_card)+1.))
            threshold[J_card] = normal.ppf(1. - (alpha/(2.*p))*t_0)

        # print( index, t_0, J, threshold)

        random_Z = np.random.standard_normal(p)
        sel = selection(X, y, random_Z)
        lam, epsilon, active, betaE, cube, initial_soln = sel
        lagrange = lam * np.ones(p)
        active_sign = np.sign(betaE)
        nactive = active.sum()

        feasible_point = np.append(1, np.fabs(betaE))

        noise_variance = 1.

        randomizer = randomization.isotropic_gaussian((p,), 1.)

        parameter = np.random.standard_normal(nactive)
        #mean = X[:, active].dot(parameter)
        mean = X[:, active].dot(np.zeros(nactive))

        test_point = np.append(np.random.uniform(low=-2.0, high=2.0, size=n), feasible_point)

        sel_prob = selection_probability_genes_variants(X,
                                                        feasible_point,
                                                        index,
                                                        J,
                                                        active,
                                                        T_sign,
                                                        active_sign,
                                                        lagrange,
                                                        threshold,
                                                        mean,
                                                        noise_variance,
                                                        randomizer,
                                                        epsilon,
                                                        coef=1.,
                                                        offset=None,
                                                        quadratic=None,
                                                        nstep=10)

        sel_prob_simes_lasso = sel_prob.minimize2(nstep=200)[::-1]
        print("selection prob and minimizer- fs", sel_prob_simes_lasso[0], sel_prob_simes_lasso[1])


#sel_prob_ms_lasso()

def valid_inference():
    n = 100
    p = 5000
    s = 0
    snr = 0.

    X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

    n, p = X.shape

    alpha = 0.05

    sel_simes = simes_selection(X, y, alpha=0.05, randomizer='gaussian')

    if sel_simes is not None:

        index = sel_simes[0]

        t_0 = sel_simes[2]

        J = sel_simes[1]

        T_sign = sel_simes[3]*np.ones(1)

        T_stats = sel_simes[4]*np.ones(1)

        if t_0 == 0:
            threshold = normal.ppf(1.- alpha/(2.*p))*np.ones(1)

        else:
            J_card = J.shape[0]
            threshold = np.zeros(J_card+1)
            threshold[:J_card] = normal.ppf(1. - (alpha/(2.*p))*(np.arange(J_card)+1.))
            threshold[J_card] = normal.ppf(1. - (alpha/(2.*p))*t_0)

        random_Z = np.random.standard_normal(p)
        sel = selection(X, y, random_Z)
        lam, epsilon, active, betaE, cube, initial_soln = sel

        if sel is not None:

            lagrange = lam * np.ones(p)
            active_sign = np.sign(betaE)
            nactive = active.sum()
            print("number of selected variables by Lasso", nactive)

            feasible_point = np.append(1, np.fabs(betaE))

            noise_variance = 1.

            randomizer = randomization.isotropic_gaussian((p,), 1.)

            generative_X = X[:, active]
            prior_variance = 100.

            grad_map = sel_prob_gradient_map_simes_lasso(X,
                                                         feasible_point,
                                                         index,
                                                         J,
                                                         active,
                                                         T_sign,
                                                         active_sign,
                                                         lagrange,
                                                         threshold,
                                                         generative_X,
                                                         noise_variance,
                                                         randomizer,
                                                         epsilon)

            #print("here", grad_map.smooth_objective(np.zeros(nactive), mode='both'))

            inf = selective_inf_simes_lasso(y, grad_map, prior_variance)

            #sel_MAP = inf.map_solve(nstep=100)[::-1]

            #print("selective MAP- simes_lasso_screening", sel_MAP[1])

            toc = time.time()
            samples = inf.posterior_samples()
            tic = time.time()

            adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
            print('sampling time', tic - toc)
            print("adjusted intervals", adjusted_intervals)

            sel_mean = np.mean(samples, axis=0)


valid_inference()