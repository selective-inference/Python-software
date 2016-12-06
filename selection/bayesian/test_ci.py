import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.approximation_based_intervals import approximate_conditional_sel_prob, \
    approximate_conditional_density
from selection.randomized.api import randomization

n = 100
p = 20
s = 5
snr = 5

sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()
random_Z = np.random.standard_normal(p)
sel = selection(X_1, y, random_Z, randomization_scale=1, sigma=None, lam=None)
lam, epsilon, active, betaE, cube, initial_soln = sel
print true_beta, active

truth = ((np.linalg.pinv(X_1[:,active])[0, :]).T).dot(X_1.dot(true_beta))

print("truth",truth)

def approximate_ci_test():

    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()

    random_Z = np.random.standard_normal(p)

    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel

    active_set = np.asarray([i for i in range(p) if active[i]])

    print("active set", active_set)

    if active[active_set[0]] == True:

        noise_variance = 1.

        active_signs = np.sign(betaE)

        tau = 1.

        lagrange = lam * np.ones(p)

        feasible_point = np.fabs(betaE)

        approx_val = approximate_conditional_density(y,
                                                     X_1,
                                                     feasible_point,
                                                     active,
                                                     active_signs,
                                                     lagrange,
                                                     noise_variance,
                                                     randomization.isotropic_gaussian((p,), tau),
                                                     epsilon,
                                                     j=0)

        ci = approx_val.approximate_ci()

        return ci

coverage =0
null = 0

for iter in range(200):

    intervals = approximate_ci_test()
    print(iter, intervals)
    if intervals is None:
        coverage = coverage
        null = null + 1
    else:
        if (intervals[0]==0) and (intervals[1]==0):
             coverage = coverage
             null = null + 1
        elif (truth >= intervals[0]) and (truth <= intervals[1]):
            coverage += 1

    print coverage

print null
print coverage/(200.-null)