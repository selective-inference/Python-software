from __future__ import print_function
import time
import numpy as np
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection, instance
from selection.randomized.api import randomization
from selection.bayesian.inference_fs import sel_prob_gradient_map_fs, selective_map_credible_fs

n = 50
p = 10
s = 0
snr = 5
sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

def test_inf_fs():

    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()
    #print(true_beta, nonzero, noise_variance)
    random_Z = np.random.standard_normal(p)

    random_obs = X_1.T.dot(y) + random_Z
    active_index = np.argmax(random_obs)
    active = np.zeros(p, bool)
    active[active_index] = 1
    active_sign = np.sign(random_obs[active_index])
    nactive = 1

    primal_feasible = np.fabs(random_obs[active_index])
    tau = 1  # randomization_variance

    parameter = np.random.standard_normal(nactive)
    mean = X_1[:, active].dot(parameter)

    generative_X = X_1[:, active]
    prior_variance = 1000.

    inf_rr = selective_map_credible_fs(y,
                                       X_1,
                                       primal_feasible,
                                       active,
                                       active_sign,
                                       generative_X,
                                       noise_variance,
                                       prior_variance,
                                       randomization.isotropic_gaussian((p,), tau))

    #map = inf_rr.map_solve_2(nstep = 100)[::-1]

    #print ("gradient at map", -inf_rr.smooth_objective(map[1], mode='grad'))
    #print ("map objective, map", map[0], map[1])
    #toc = time.time()
    samples = inf_rr.posterior_samples()
    #tic = time.time()
    #print('sampling time', tic - toc)
    return samples

coverage = 0
for i in range(100):
    post_samples = test_inf_fs()
    print("iteration", i)
    lc = np.percentile(post_samples, 5, axis=0)
    uc = np.percentile(post_samples, 95, axis=0)
    print(lc, uc)
    if lc< 0. and 0.< uc:
        coverage+=1
    else:
        coverage= coverage

    print(coverage)

