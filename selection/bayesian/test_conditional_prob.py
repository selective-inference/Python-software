from __future__ import print_function
import time

import numpy as np
import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.ci_intervals_approx import approximate_conditional_prob, approximate_conditional_density
from selection.randomized.api import randomization
from selection.bayesian.paired_bootstrap import pairs_bootstrap_glm, bootstrap_cov

n = 100
p = 20
s = 3
snr = 5

sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()
random_Z = np.random.standard_normal(p)
sel = selection(X_1, y, random_Z, randomization_scale=1, sigma=None, lam=None)
lam, epsilon, active, betaE, cube, initial_soln = sel
print(true_beta, active)

truth = ((np.linalg.pinv(X_1[:,active])[0, :]).T).dot(X_1.dot(true_beta))

print("truth",truth)

def test_approximate_conditional_prob():

    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()

    random_Z = np.random.standard_normal(p)

    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel

    active_set = np.asarray([i for i in range(p) if active[i]])

    nactive = active.sum()

    active_signs = np.sign(betaE)

    lagrange = lam * np.ones(p)

    print("active set", active_set)

    bootstrap_score = pairs_bootstrap_glm(rr.glm.gaussian(X_1, y), active, beta_full=None, inactive=~active)[0]
    sampler = lambda: np.random.choice(n, size=(n,), replace=True)
    cov = bootstrap_cov(sampler, bootstrap_score)

    #arguments to be given are : target, A, null_statistic

    Sigma_D_T = cov[:,:nactive]
    Sigma_T_inv = np.linalg.inv(cov[:nactive, :nactive])

    X_active = X_1[:, active]
    B = X_1.T.dot(X_active)

    B_active = B[active]
    B_nactive = B[~active]

    data_active = np.hstack([-B_active, np.zeros((nactive, p - nactive))])
    data_nactive = np.hstack([-B_nactive, np.identity(p - nactive)])
    data_coef = np.vstack([data_active, data_nactive])

    A = (data_coef.dot(Sigma_D_T)).dot(Sigma_T_inv)
    print("shape of A", np.shape(A))

    #observed target and null statistic
    X_inactive = X_1[:, ~active]
    X_gen_inv = np.linalg.pinv(X_active)
    X_projection = X_active.dot(X_gen_inv)
    X_inter = (X_inactive.T).dot((np.identity(n) - X_projection))
    D_mean = np.vstack([X_gen_inv, X_inter])
    data_obs = D_mean.dot(y)
    target_obs = data_obs[:nactive]
    null_statistic = (data_coef.dot(data_obs) - A.dot(target_obs))
    feasible_point = np.fabs(betaE)

    approx_cond = approximate_conditional_prob(X_1,
                                               target_obs,
                                               A, # the coef matrix of target
                                               null_statistic, #null statistic that stays fixed
                                               feasible_point,
                                               active,
                                               active_signs,
                                               lagrange,
                                               randomization.isotropic_gaussian((p,), 1.),
                                               epsilon,
                                               t= 3.)
    #test_point = np.fabs(betaE)
    j_ind = 1
    #cond_density = approx_cond.sel_prob(j_ind)
    #cond_density = approx_cond.sel_prob_smooth_objective(test_point, j_ind, mode= 'func')
    cond_density = approx_cond.minimize2(j_ind)

    print("conditional density", cond_density)

#test_approximate_conditional_prob()


def test_approximate_ci():

    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()

    random_Z = np.random.standard_normal(p)

    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel

    active_set = np.asarray([i for i in range(p) if active[i]])

    nactive = active.sum()

    active_signs = np.sign(betaE)

    lagrange = lam * np.ones(p)

    print("active set", active_set)

    bootstrap_score = pairs_bootstrap_glm(rr.glm.gaussian(X_1, y), active, beta_full=None, inactive=~active)[0]
    sampler = lambda: np.random.choice(n, size=(n,), replace=True)
    cov = bootstrap_cov(sampler, bootstrap_score)

    feasible_point = np.fabs(betaE)

    approximate_den = approximate_conditional_density(y,
                                                      X_1,
                                                      feasible_point,
                                                      active,
                                                      active_signs,
                                                      lagrange,
                                                      cov,
                                                      noise_variance,
                                                      randomization.isotropic_gaussian((p,), 1.),
                                                      epsilon)

    ci = approximate_den.approximate_ci(1)
    print("approximate ci", ci)

test_approximate_ci()









