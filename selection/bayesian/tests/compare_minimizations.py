import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.bayesian.sel_probability import selection_probability
from selection.bayesian.non_scaled_sel_probability import no_scale_selection_probability
from selection.bayesian.sel_probability2 import cube_subproblem, cube_gradient, cube_barrier, selection_probability_objective
from selection.bayesian.dual_optimization import dual_selection_probability
from selection.randomized.api import randomization
from selection.bayesian.selection_probability import selection_probability_methods

#fixing n, p, true sparsity and signal strength
n = 10
p = 3
s = 1
snr = 5

#sampling the Gaussian instance
X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
random_Z = np.random.standard_normal(p)
#getting randomized Lasso solution
sel = selection(X_1,y, random_Z)

lam, epsilon, active, betaE, cube, initial_soln = sel
print epsilon, lam, betaE
noise_variance = 1
nactive=betaE.shape[0]
active_signs = np.sign(betaE)
tau=1 #randomization_variance


def test_minimizations():
    if nactive > 1:
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        sel_prob_scipy = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                       noise_variance, tau, epsilon)
        sel_prob_scipy_val = sel_prob_scipy.minimize_scipy()
        sel_prob_grad_descent = selection_probability_objective(X_1, np.fabs(betaE), active, active_signs, lagrange,
                                                                mean,noise_variance,
                                                                randomization.isotropic_gaussian((p,), 1.),epsilon)

        print "value and minimizer- scipy", -sel_prob_scipy_val[0], sel_prob_scipy_val[1]

        print "value and minimizer- grad descent", -sel_prob_grad_descent.minimize()[1], sel_prob_grad_descent.minimize()[0]

test_minimizations()

def one_sparse_minimizations():
    if nactive == 1:
        snr_seq = np.linspace(-10, 10, num=100)
        lagrange = lam * np.ones(p)
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            print "parameter value", parameter
            mean = X_1[:, active].dot(parameter)

            sel_prob_scipy = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                           noise_variance, tau, epsilon)

            sel_prob_scipy_val = sel_prob_scipy.minimize_scipy()

            sel_prob_grad_descent = selection_probability_objective(X_1, np.fabs(betaE), active, active_signs, lagrange,
                                                                    mean,
                                                                    noise_variance,
                                                                    randomization.isotropic_gaussian((p,), 1.),
                                                                    epsilon)

            print "value and minimizer- scipy", -sel_prob_scipy_val[0], sel_prob_scipy_val[1]

            print "value and minimizer- grad descent", -sel_prob_grad_descent.minimize()[1], \
            sel_prob_grad_descent.minimize()[0]

#one_sparse_minimizations()

def test_objectives_one_sparse():
    if nactive == 1:
        snr_seq = np.linspace(-10, 10, num=100)
        lagrange = lam * np.ones(p)
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            print "parameter value", parameter
            mean = X_1[:, active].dot(parameter)
            vec = np.random.standard_normal(n)
            active_coef = np.dot(np.diag(active_signs), np.fabs(np.random.standard_normal(nactive)))
            sel_prob_scipy = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                           noise_variance, tau, epsilon)

            sel_scipy_objective = sel_prob_scipy.objective(np.append(vec, np.fabs(active_coef)))

            sel_prob_grad_descent = selection_probability_objective(X_1, np.fabs(betaE), active, active_signs, lagrange,
                                                                    mean,
                                                                    noise_variance,
                                                                    randomization.isotropic_gaussian((p,), 1.),
                                                                    epsilon)

            sel_grad_objective = sel_prob_grad_descent.smooth_objective(np.append(vec, np.fabs(active_coef)),
                                                                        mode='func', check_feasibility=False)

            print "objective - new function", sel_scipy_objective
            print "objective - to be debugged", sel_grad_objective

#test_objectives_one_sparse()

def test_objectives_not_one_sparse():
    if nactive > 1:
        parameter = np.random.standard_normal(nactive)
        print "parameter value", parameter
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        vec = np.random.standard_normal(n)
        active_coef = np.dot(np.diag(active_signs), np.fabs(np.random.standard_normal(nactive)))

        sel_prob_scipy = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                       noise_variance, tau, epsilon)

        sel_scipy_objective = sel_prob_scipy.objective(np.append(vec, np.fabs(active_coef)))

        sel_prob_grad_descent = selection_probability_objective(X_1, np.fabs(betaE), active, active_signs, lagrange,
                                                                mean,
                                                                noise_variance,
                                                                randomization.isotropic_gaussian((p,), 1.),
                                                                epsilon)

        sel_grad_objective = sel_prob_grad_descent.smooth_objective(np.append(vec, np.fabs(active_coef)),
                                                                    mode='func', check_feasibility=False)

        print "objective - for scipy.optimize", sel_scipy_objective
        print "objective - for grad descent", sel_grad_objective

#test_objectives_not_one_sparse()









