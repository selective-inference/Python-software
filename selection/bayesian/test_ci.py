import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.approximation_based_intervals import approximate_conditional_sel_prob, \
    approximate_conditional_density
from selection.randomized.api import randomization

n = 15
p = 5
s = 3
snr = 5
sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()

random_Z = np.random.standard_normal(p)

sel = selection(X_1, y, random_Z)

lam, epsilon, active, betaE, cube, initial_soln = sel

print true_beta, active

truth = ((np.linalg.pinv(X_1[:,active])[1, :]).T).dot(X_1[:,active].dot(betaE))

print truth

def approximate_ci_test():

    X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()

    random_Z = np.random.standard_normal(p)

    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel

    if active[1] == True:

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
                                                     j=1)

        ci = approx_val.approximate_ci()
        lci = ci[0]
        uci = ci[1]

        print(lci, uci)

    return lci, uci

coverage =0

for iter in range(100):

    print iter

    lc, uc = approximate_ci_test()

    if (truth >= lc) and (truth <= uc):
        coverage += 1

print coverage