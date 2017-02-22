from __future__ import print_function
import time

import os, numpy as np, pandas, statsmodels.api as sm
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.bayesian.cisEQTLS.inference_2sels import selection_probability_genes_variants, \
    sel_prob_gradient_map_simes_lasso, selective_inf_simes_lasso
from scipy.stats import norm as normal
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection, BH_q

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
    n = 350
    p = 200
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


#valid_inference()

def test_data_file():

    gene_1 = pandas.read_table("/Users/snigdhapanigrahi/tiny_example/1.txt", na_values="NA")
    X = np.array(gene_1.ix[:, 2:])
    n, p = X.shape

    y = np.sqrt(n) * np.array(gene_1.ix[:, 1])

    alpha = 0.05

    sel_simes = simes_selection(X, y, alpha=0.05, randomizer='gaussian')

    if sel_simes is not None:

        index = sel_simes[0]

        t_0 = sel_simes[2]

        J = sel_simes[1]

        T_sign = sel_simes[3] * np.ones(1)

        T_stats = sel_simes[4] * np.ones(1)

        if t_0 == 0:
            threshold = normal.ppf(1. - alpha / (2. * p)) * np.ones(1)

        else:
            J_card = J.shape[0]
            threshold = np.zeros(J_card + 1)
            threshold[:J_card] = normal.ppf(1. - (alpha / (2. * p)) * (np.arange(J_card) + 1.))
            threshold[J_card] = normal.ppf(1. - (alpha / (2. * p)) * t_0)

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

            inf = selective_inf_simes_lasso(y, grad_map, prior_variance)

            # sel_MAP = inf.map_solve(nstep=100)[::-1]

            # print("selective MAP- simes_lasso_screening", sel_MAP[1])

            toc = time.time()
            samples = inf.posterior_samples()
            tic = time.time()

            Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
            post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
            post_var = prior_variance * np.identity(nactive) - ((prior_variance ** 2) * (generative_X.T.dot(Q).dot(generative_X)))
            unadjusted_intervals = np.vstack([post_mean - 1.65 * (post_var.diagonal()), post_mean + 1.65 * (post_var.diagonal())])


            adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
            print('sampling time', tic - toc)
            print("unadjusted intervals", unadjusted_intervals)
            print("adjusted intervals", adjusted_intervals)

            sel_mean = np.mean(samples, axis=0)
            print("unadjusted mean", post_mean)
            print("selective mean", sel_mean)


#test_data_file()

def p_value():
    n = 350
    p = 100
    s = 10
    snr = 5.

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
            prior_variance = 1000.

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

            ngrid = 1000
            Q = np.linalg.inv(prior_variance * (generative_X.dot(generative_X.T)) + noise_variance * np.identity(n))
            post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))

            unad_pval = 2* np.minimum(1.-normal.cdf(np.abs(post_mean)), normal.cdf(-np.abs(post_mean)))


            quantiles = np.zeros((ngrid, nactive))
            print("shape", quantiles.shape)
            for i  in range(ngrid):
                quantiles[i,:] = np.percentile(samples, (i*100.)/ngrid, axis=0)

            index_grid = np.argmin(np.abs(quantiles - np.zeros((ngrid, nactive))), axis = 0)
            print(index_grid)
            p_value = 2 * np.minimum(np.true_divide(index_grid,ngrid),1.- np.true_divide(index_grid,ngrid))
            print(np.sort(p_value))

            adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
            print("adjusted intervals", adjusted_intervals)

            p_BH = BH_q(p_value, 0.05)

            if p_BH is not None:
                print("results from BH", p_BH[0], p_BH[1])

            print("unadjusted p-values", unad_pval)

            p_BH_unad = BH_q(unad_pval , 0.05)

            if p_BH_unad is not None:
                print("results from BH", p_BH_unad[0], p_BH_unad[1])

p_value()