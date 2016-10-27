from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection
from selection.bayesian.selection_probability_rr import cube_subproblem, cube_gradient, cube_barrier,\
    cube_subproblem_scaled, cube_gradient_scaled, cube_barrier_scaled, cube_subproblem_scaled,\
    selection_probability_objective
from selection.randomized.api import randomization
from selection.bayesian.dual_optimization import selection_probability_dual_objective

def one_sparse_minimizations():
    # fixing n, p, true sparsity and signal strength
    n = 10
    p = 3
    s = 1
    snr = 5

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(lam, betaE)
    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(betaE)
    primal_feasible = np.fabs(betaE)

    if nactive == 1:
        snr_seq = np.linspace(0, 20, num=5)
        #snr_seq = np.hstack([snr_seq[:25], snr_seq[25:][::-1]])
        #snr_seq = snr_seq[25:][::-1]
        lagrange = lam * np.ones(p)
        result = []
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            mean = X_1[:, active].dot(parameter)
            primal_regreg = selection_probability_objective(X_1,
                                                            primal_feasible,
                                                            active,
                                                            active_signs,
                                                            lagrange,
                                                            mean,
                                                            noise_variance,
                                                            randomization.isotropic_gaussian((p,), 1.),
                                                            epsilon)


            primal_val = primal_regreg.minimize(max_its=1000, min_its=500, tol=1.e-12)[::-1]
            primal_sol = primal_val[1]

            dual_regreg = selection_probability_dual_objective(X_1,
                                                               dual_feasible,
                                                               active,
                                                               active_signs,
                                                               lagrange,
                                                               mean,
                                                               noise_variance,
                                                               randomization.isotropic_gaussian((p,), tau),
                                                               epsilon)

            dual_val = dual_regreg.minimize(max_its=2000, min_its=1000, tol=1.e-12)[::-1]
            dual_sol = mean - (dual_regreg.X_permute.dot(np.linalg.inv(dual_regreg.B_p.T))).dot(dual_val[1])

            print(parameter, -primal_val[0], dual_val[0])

            #result.append([parameter,-primal_val[0],dual_val[0]])

        #return np.array(result)

one_sparse_minimizations()



