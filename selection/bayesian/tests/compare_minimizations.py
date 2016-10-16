from __future__ import print_function

import numpy as np
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection
from selection.bayesian.sel_probability import selection_probability
from selection.bayesian.non_scaled_sel_probability import no_scale_selection_probability
from selection.bayesian.selection_probability_rr import cube_subproblem, cube_gradient, cube_barrier, selection_probability_objective
from selection.bayesian.dual_optimization import dual_selection_probability
from selection.randomized.api import randomization
from selection.bayesian.selection_probability import selection_probability_methods

@wait_for_return_value()
def test_minimizations():

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
    print(epsilon, lam, betaE)
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance

    if nactive > 1:
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        sel_prob_scipy = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                       noise_variance, tau, epsilon)
        sel_prob_scipy_val = sel_prob_scipy.minimize_scipy()
        sel_prob_grad_descent = selection_probability_objective(X_1, 
                                                                np.fabs(betaE), 
                                                                active, 
                                                                active_signs, 
                                                                lagrange,
                                                                mean, 
                                                                noise_variance,
                                                                randomization.isotropic_gaussian((p,), 1.),
                                                                epsilon)

        _scipy = [sel_prob_scipy_val[0], sel_prob_scipy_val[1]]
        _regreg = sel_prob_grad_descent.minimize()[::-1]

        obj1 = sel_prob_scipy.objective
        obj2 = lambda x : sel_prob_grad_descent.smooth_objective(x, 'func')
        print("value and minimizer- scipy", _scipy, obj1(_scipy[1]), obj2(_scipy[1]))
        print("value and minimizer- regreg", _regreg, obj1(_regreg[1]), obj2(_regreg[1]))
        return _scipy[0], _regreg[0]

@wait_for_return_value()
def test_one_sparse_minimizations():

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
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance

    if nactive == 1:
        snr_seq = np.linspace(-10, 10, num=6)
        lagrange = lam * np.ones(p)
        result = []
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            mean = np.squeeze(X_1[:, active].dot(parameter)) # make sure it is vector

            sel_prob_scipy = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                           noise_variance, tau, epsilon)

            sel_prob_scipy_val = sel_prob_scipy.minimize_scipy()

            sel_prob_grad_descent = selection_probability_objective(X_1, 
                                                                    np.fabs(betaE), 
                                                                    active, 
                                                                    active_signs, 
                                                                    lagrange,
                                                                    mean,
                                                                    noise_variance,
                                                                    randomization.isotropic_gaussian((p,), tau),
                                                                    epsilon)

            obj1 = sel_prob_scipy.objective
            obj2 = lambda x : sel_prob_grad_descent.smooth_objective(x, 'func')
            obj3 = lambda x : sel_prob_grad_descent.objective(x)

            _scipy = [sel_prob_scipy_val[0], sel_prob_scipy_val[1]]
            _regreg = sel_prob_grad_descent.minimize()[::-1]

            result.append([obj1(_scipy[1]), obj2(_scipy[1]), obj3(_scipy[1]), 
                           obj1(_regreg[1]), obj2(_regreg[1]), obj3(_regreg[1])])

        return np.array(result)

@wait_for_return_value()
def test_individual_terms():

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
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance

    if nactive == 1:
        snr_seq = np.linspace(-10, 10, num=6)
        lagrange = lam * np.ones(p)
        result = []
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            mean = X_1[:, active].dot(parameter)

            sel_prob_scipy = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                           noise_variance, tau, epsilon)

            sel_prob_scipy_val = sel_prob_scipy.minimize_scipy()

            sel_prob_grad_descent = selection_probability_objective(X_1, 
                                                                    np.fabs(betaE), 
                                                                    active, 
                                                                    active_signs, 
                                                                    lagrange,
                                                                    mean,
                                                                    noise_variance,
                                                                    randomization.isotropic_gaussian((p,), tau),
                                                                    epsilon)

            _scipy = [sel_prob_scipy_val[0], sel_prob_scipy_val[1]]
            _regreg = sel_prob_grad_descent.minimize()[::-1]

            # the _regreg solution is feasible

            for param in [_scipy[1], _regreg[1]]:
                np.testing.assert_allclose(sel_prob_scipy.likelihood(param),
                                           sel_prob_grad_descent.likelihood_loss.smooth_objective(param, 'func'))

                np.testing.assert_allclose(sel_prob_scipy.cube_problem(param, method='softmax_barrier'),
                                           sel_prob_grad_descent.cube_objective(param)[0], rtol=1.e-5)

                np.testing.assert_allclose(sel_prob_scipy.active_conjugate_objective(param),
                                           sel_prob_grad_descent.active_conjugate_objective(param)[0], rtol=1.e-5)

                np.testing.assert_allclose(sel_prob_scipy.nonneg(param),
                                           sel_prob_grad_descent.nonnegative_barrier.smooth_objective(param, 'func'))

                # check the objective is the sum of terms it's supposed to be

                np.testing.assert_allclose(sel_prob_scipy.objective(param),
                                           sel_prob_scipy.likelihood(param) +
                                           sel_prob_scipy.cube_problem(param, method='softmax_barrier') +
                                           sel_prob_scipy.active_conjugate_objective(param) +
                                           sel_prob_scipy.nonneg(param), rtol=1.e-5)

                # check the objective values

                np.testing.assert_allclose(sel_prob_scipy.objective(param),
                                           sel_prob_grad_descent.smooth_objective(param, 'func'), rtol=1.e-5)
            
                np.testing.assert_allclose(sel_prob_scipy.objective(param),
                                           sel_prob_grad_descent.likelihood_loss.smooth_objective(param, 'func') + 
                                           sel_prob_grad_descent.cube_objective(param)[0] + 
                                           sel_prob_grad_descent.active_conjugate_objective(param)[0] + 
                                           sel_prob_grad_descent.nonnegative_barrier.smooth_objective(param, 'func'), rtol=1.e-5)


        return np.array(result)

@wait_for_return_value()
def test_objectives_one_sparse():

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
    print(epsilon, lam, betaE)
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance


    if nactive == 1:
        snr_seq = np.linspace(-10, 10, num=100)
        lagrange = lam * np.ones(p)
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            print("parameter value", parameter)
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

            print("objective - new function", sel_scipy_objective)
            print("objective - to be debugged", sel_grad_objective)
        return True

@wait_for_return_value()
def test_objectives_not_one_sparse():

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
    print(epsilon, lam, betaE)
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance

    if nactive > 1:
        parameter = np.random.standard_normal(nactive)
        print("parameter value", parameter)
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

        print("objective - for scipy.optimize", sel_scipy_objective)
        print("objective - for grad descent", sel_grad_objective)
        return True










