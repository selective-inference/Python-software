from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.bayesian.forward_step import cube_subproblem_fs, cube_objective_fs, selection_probability_objective_fs
from selection.randomized.api import randomization
from selection.bayesian.forward_step_reparametrized import cube_subproblem_fs_linear, cube_objective_fs_linear, \
    selection_probability_objective_fs_rp

def fs_primal_test():
    n = 30
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

    fs_rp = selection_probability_objective_fs_rp(X_1,
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

    test = np.append(y,1.)
    print(fs_rp.smooth_objective(test, mode='both'), fs.smooth_objective(test, mode='both'))

    toc = time.time()
    sel_prob_fs_rp = fs_rp.minimize2(nstep=50)[::-1]
    tic = time.time()
    print('fs_rp time', tic - toc)

    print("selection prob and minimizer- fs", sel_prob_fs, sel_prob_fs_rp)

fs_primal_test()


def fs_one_sparse_test():
    n = 50
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
    randomizer = randomization.isotropic_gaussian((p,), 1.)

    snr_seq = np.linspace(-10, 20, num=10)
    result = []
    for i in range(snr_seq.shape[0]):
        parameter = snr_seq[i]
        mean = X_1[:, active].dot(parameter)

        fs = selection_probability_objective_fs_rp(X_1,
                                                   feasible_point,
                                                   active,
                                                   active_sign,
                                                   mean,  # in R^n
                                                   noise_variance,
                                                   randomizer)


        sel_prob_fs = fs.minimize2(nstep = 50)[::-1]
        print(parameter, sel_prob_fs[0], sel_prob_fs[1])
        result.append([parameter, -sel_prob_fs[0]])

        sel_prob_fs_rp = fs.minimize2(nstep=50)[::-1]
        print(parameter, sel_prob_fs[0], sel_prob_fs[1])
        result.append([parameter, -sel_prob_fs[0]])

    return np.array(result)

#fs_one_sparse_test()