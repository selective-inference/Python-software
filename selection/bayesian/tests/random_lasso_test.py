from __future__ import print_function
import time

import numpy as np
import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.randomized.api import randomization
from selection.bayesian.paired_bootstrap import pairs_bootstrap_glm, bootstrap_cov
from selection.bayesian.randomX_lasso_primal import selection_probability_objective_randomX

def test_approximation_randomlasso():
    n = 100
    p = 20
    s = 5
    snr = 5

    sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()
    random_Z = np.random.standard_normal(p)
    sel = selection(X_1, y, random_Z, randomization_scale=1, sigma=None, lam=None)
    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(true_beta, active)

    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    bootstrap_score = pairs_bootstrap_glm(rr.glm.gaussian(X_1,y), active, beta_full=None, inactive = ~active)[0]
    sampler = lambda: np.random.choice(n, size=(n,),replace = True)
    cov = bootstrap_cov(sampler, bootstrap_score)

    X_active = X_1[:, active]
    X_inactive = X_1[:,~active]
    X_gen_inv = np.linalg.pinv(X_active)
    X_projection = X_active.dot(X_gen_inv)
    parameter = np.random.standard_normal(nactive)
    lagrange = lam * np.ones(p)

    #selected model mean
    generative_mean = X_active.dot(parameter)
    mean_suff = X_gen_inv.dot(generative_mean)
    mean_nuisance = ((X_inactive.T).dot((np.identity(n)- X_projection))).dot(generative_mean)
    mean = np.append(mean_suff, mean_nuisance)
    print("means", mean_suff, mean_nuisance)

    sel_prob_regreg = selection_probability_objective_randomX(X_1,
                                                              np.fabs(betaE),
                                                              active,
                                                              active_signs,
                                                              lagrange,
                                                              mean,
                                                              cov,
                                                              noise_variance,
                                                              randomization.isotropic_gaussian((p,), 1.),
                                                              epsilon)

    toc = time.time()
    regreg = sel_prob_regreg.minimize2(nstep=20)[::-1]
    tic = time.time()
    print('computation time', tic - toc)
    print('selection prob', regreg[0])

test_approximation_randomlasso()




