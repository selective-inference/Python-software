from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.bayesian.forward_step import cube_subproblem_fs, cube_objective_fs, selection_probability_objective_fs
from selection.randomized.api import randomization

def fs_primal_test():
    n = 100
    p = 10
    s = 5
    snr = 3

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    print(true_beta, nonzero, noise_variance)
    random_Z = np.random.standard_normal(p)
    random_obs = X_1.T.dot(y) + random_Z
    active_index = np.argmax(random_obs)
    active = np.zeros(p, bool)
    active[active_index] = 1
    active_sign = np.sign(random_obs[active_index])
    nactive = 1


    feasible_point = np.fabs(random_obs[active_index])
    tau = 1 #randomization_variance

    parameter = np.random.standard_normal(nactive)
    mean = X_1[:, active].dot(parameter)

    #print(active_index, active, active_sign, mean)
    randomizer = randomization.isotropic_gaussian((p,), 1.)
    cube = cube_objective_fs(randomizer.CGF_conjugate)

    test = np.ones(p)

    print(cube.smooth_objective(test))
    fs = selection_probability_objective_fs(X_1,
                                            feasible_point,
                                            active,
                                            active_sign,
                                            mean,  # in R^n
                                            noise_variance,
                                            randomizer)

    toc = time.time()
    sel_prob_fs = fs.minimize2(nstep = 50)[::-1]
    tic = time.time()
    print('fs time', tic-toc)

    print("selection prob and minimizer- fs", sel_prob_fs)

fs_primal_test()
