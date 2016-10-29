from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection
from selection.bayesian.selection_probability_rr import cube_subproblem, cube_gradient, cube_barrier, \
    selection_probability_objective, cube_subproblem_scaled, cube_gradient_scaled, cube_barrier_scaled, \
    cube_subproblem_scaled
from selection.randomized.api import randomization
from selection.bayesian.selection_probability import selection_probability_methods
from selection.bayesian.dual_scipy import dual_selection_probability_func
from selection.bayesian.dual_optimization import selection_probability_dual_objective


#@wait_for_return_value()
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

        sel_prob_scipy = selection_probability_methods(X_1,
                                                       np.fabs(betaE),
                                                       active,
                                                       active_signs,
                                                       lagrange,
                                                       mean,
                                                       noise_variance,
                                                       tau,
                                                       epsilon)
        toc = time.time()
        sel_prob_scipy_val = sel_prob_scipy.minimize_scipy()
        tic = time.time()
        print('scipy time', tic-toc)

        sel_prob_grad_descent = selection_probability_objective(X_1,
                                                                np.fabs(betaE),
                                                                active,
                                                                active_signs,
                                                                lagrange,
                                                                mean,
                                                                noise_variance,
                                                                randomization.isotropic_gaussian((p,), 1.),
                                                                epsilon,
                                                                nstep=10)

        _scipy = [sel_prob_scipy_val[0], sel_prob_scipy_val[1]]
        toc = time.time()
        _regreg = sel_prob_grad_descent.minimize(max_its=100, tol=1.e-12)[::-1]
        tic = time.time()
        print('regreg time', tic-toc)

        obj1 = sel_prob_scipy.objective
        obj2 = lambda x : sel_prob_grad_descent.smooth_objective(x, 'func')

        toc = time.time()
        _regreg2 = sel_prob_grad_descent.minimize2(nstep=20)[::-1]
        tic = time.time()
        print('regreg2', tic-toc)


        print("value and minimizer- scipy", _scipy, obj1(_scipy[1]), obj2(_scipy[1]))
        print("value and minimizer- regreg", _regreg, obj1(_regreg[1]), obj2(_regreg[1]))
        return _scipy[0], _regreg[0]

#@wait_for_return_value()
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
        snr_seq = np.linspace(-10, 10, num=51)
        snr_seq = np.hstack([snr_seq[25:], snr_seq[:25][::-1]])
        lagrange = lam * np.ones(p)
        result = []
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            mean = np.squeeze(X_1[:, active].dot(parameter)) # make sure it is vector

            sel_prob_scipy = selection_probability_methods(X_1,
                                                           np.fabs(betaE),
                                                           active,
                                                           active_signs,
                                                           lagrange,
                                                           mean,
                                                           noise_variance,
                                                           tau,
                                                           epsilon)

            toc = time.time()
            sel_prob_scipy_val = sel_prob_scipy.minimize_scipy()
            tic = time.time()
            print('scipy time', tic-toc)

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

            toc = time.time()
            _regreg = sel_prob_grad_descent.minimize(min_its=500, max_its=1000, tol=1.e-12)[::-1]
            tic = time.time()
            print('regreg time', tic-toc)

            toc = time.time()
            _regreg2 = sel_prob_grad_descent.minimize2()[::-1]
            tic = time.time()
            print('regreg2 time', tic-toc)

            obj1 = sel_prob_scipy.objective
            obj2 = lambda x : sel_prob_grad_descent.smooth_objective(x, 'func')
            obj3 = lambda x : sel_prob_grad_descent.objective(x)

            #check_two_approaches(_scipy[1], sel_prob_scipy, sel_prob_grad_descent)
            #check_two_approaches(_regreg[1], sel_prob_scipy, sel_prob_grad_descent)

            result.append([obj1(_scipy[1]), obj2(_scipy[1]), obj3(_scipy[1]),
                           obj1(_regreg[1]), obj2(_regreg[1]), obj3(_regreg[1]),
                           obj1(_regreg2[1]), obj2(_regreg2[1]), obj3(_regreg2[1])])

            print('scipy gradient', sel_prob_grad_descent.smooth_objective(_scipy[1], 'grad'))
            print('regreg gradient', sel_prob_grad_descent.smooth_objective(_regreg[1], 'grad'))

        return np.array(result)

#@wait_for_return_value()
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

                check_two_approaches(param, sel_prob_scipy, sel_prob_grad_descent)


        return np.array(result)

def check_two_approaches(param, sel_prob_scipy, sel_prob_grad_descent):

    np.testing.assert_allclose(sel_prob_scipy.likelihood(param),
                               sel_prob_grad_descent.likelihood_loss.smooth_objective(param, 'func'))

    np.testing.assert_allclose(sel_prob_scipy.cube_problem(param, method='softmax_barrier'),
                               sel_prob_grad_descent.cube_loss.smooth_objective(param, 'func'), rtol=1.e-5)

    np.testing.assert_allclose(sel_prob_scipy.active_conjugate_objective(param),
                               sel_prob_grad_descent.active_conj_loss.smooth_objective(param, 'func'))

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
                               sel_prob_grad_descent.cube_loss.smooth_objective(param, 'func') +
                               sel_prob_grad_descent.active_conj_loss.smooth_objective(param, 'func') +
                               sel_prob_grad_descent.nonnegative_barrier.smooth_objective(param, 'func'), rtol=1.e-5)



#@wait_for_return_value()
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

#@wait_for_return_value()
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


################check for dual
def check_duals(param, dual_scipy, dual_regreg):

    np.testing.assert_allclose(dual_scipy.data_CGF(param),
                               dual_regreg.likelihood_loss.smooth_objective(param, 'func'))

    np.testing.assert_allclose(dual_scipy.rand_CGF(param),
                               dual_regreg.CGF_randomizer.smooth_objective(param, 'func'), rtol=1.e-5)

    np.testing.assert_allclose(dual_scipy.composed_barrier_conjugate(param),
                               dual_regreg.conjugate_barrier.smooth_objective(param, 'func'))

    np.testing.assert_allclose(dual_scipy.linear_term(param),
                               dual_regreg.linear_term.objective(param, 'func'))

    # check the objective is the sum of terms it's supposed to be

    np.testing.assert_allclose(dual_scipy.dual_objective(param),
                               dual_scipy.data_CGF(param) +
                               dual_scipy.rand_CGF(param) +
                               dual_scipy.composed_barrier_conjugate(param) +
                               dual_scipy.linear_term(param), rtol=1.e-5)

    # check the objective values

    np.testing.assert_allclose(dual_scipy.dual_objective(param),
                               dual_regreg.total_loss.objective(param, 'func'), rtol=1.e-5)

    np.testing.assert_allclose(dual_scipy.dual_objective(param),
                               dual_regreg.likelihood_loss.smooth_objective(param, 'func') +
                               dual_regreg.CGF_randomizer.smooth_objective(param, 'func') +
                               dual_regreg.conjugate_barrier.smooth_objective(param, 'func') +
                               dual_regreg.linear_term.objective(param, 'func'), rtol=1.e-5)


def test_individual_terms_dual():

    #fixing n, p, true sparsity and signal strength
    n = 10
    p = 20
    s = 5
    snr = 5

    #sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    #getting randomized Lasso solution
    sel = selection(X_1,y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(betaE)
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))

    if nactive == 1:
        snr_seq = np.linspace(-10, 10, num=6)
        lagrange = lam * np.ones(p)
        result = []
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            mean = X_1[:, active].dot(parameter)

            dual_scipy = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                           noise_variance, tau, epsilon)


            dual_regreg = selection_probability_dual_objective(X_1,
                                                               dual_feasible,
                                                               active,
                                                               active_signs,
                                                               lagrange,
                                                               mean,
                                                               noise_variance,
                                                               randomization.isotropic_gaussian((p,), tau),
                                                               epsilon)

            test_point = np.ones(p)
            test_point[:nactive] = -np.fabs(np.random.standard_normal(nactive))

            check_duals(test_point, dual_scipy, dual_regreg)

    else:
        for i in range(6):
            parameter = np.random.standard_normal(nactive)
            lagrange = lam * np.ones(p)
            mean = X_1[:, active].dot(parameter)
            dual_scipy = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                         noise_variance, tau, epsilon)

            dual_regreg = selection_probability_dual_objective(X_1,
                                                               dual_feasible,
                                                               active,
                                                               active_signs,
                                                               lagrange,
                                                               mean,
                                                               noise_variance,
                                                               randomization.isotropic_gaussian((p,), tau),
                                                               epsilon)
            test_point = np.zeros(p)
            test_point[:nactive] = -np.fabs(np.random.standard_normal(nactive))
            test_point[nactive:] = np.random.standard_normal(p-nactive)
            check_duals(test_point, dual_scipy, dual_regreg)

#test_individual_terms_dual()

def test_dual_minimizations():

    #fixing n, p, true sparsity and signal strength
    n = 30
    p = 20
    s = 3
    snr = 5

    #sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    #getting randomized Lasso solution
    sel = selection(X_1,y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    #print(epsilon, lam, betaE)
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
    #print('loc1')

    if nactive > 1:
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)

        dual_scipy = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                     noise_variance, tau, epsilon)

        dual_regreg = selection_probability_dual_objective(X_1,
                                                           dual_feasible,
                                                           active,
                                                           active_signs,
                                                           lagrange,
                                                           mean,
                                                           noise_variance,
                                                           randomization.isotropic_gaussian((p,), tau),
                                                           epsilon)

        dual_regreg2 = selection_probability_dual_objective(X_1,
                                                            dual_feasible,
                                                            active,
                                                            active_signs,
                                                            lagrange,
                                                            mean,
                                                            noise_variance,
                                                            randomization.isotropic_gaussian((p,), tau),
                                                            epsilon)
        toc = time.time()
        #print (dual_feasible, 'dual2')
        dual_scipy_val = dual_scipy.minimize_dual()
        tic = time.time()
        print('scipy time', tic-toc)

        _scipy = [dual_scipy_val[0], dual_scipy_val[1]]

        toc = time.time()
        _regreg = dual_regreg.minimize(max_its=2000, min_its=1000, tol=1.e-12)[::-1]
        tic = time.time()
        print('regreg time', tic-toc)
        #print (dual_feasible, 'dual')

        obj1 = dual_scipy.dual_objective
        obj2 = lambda x : dual_regreg.total_loss.objective(x, 'func')

        grad = lambda x: (dual_regreg.total_loss.smooth_objective(x, 'grad') + dual_regreg.dual_arg)

        toc = time.time()
        _regreg2 = dual_regreg2.minimize2(nstep=100)[::-1]
        tic = time.time()
        print('regreg2', tic-toc)

        #print (dual_feasible, 'dual3')

        print("value and minimizer- scipy", _scipy, obj1(_scipy[1]), obj2(_scipy[1]))
        print("value and minimizer- regreg", _regreg, obj1(_regreg[1]), obj2(_regreg[1]))
        print("value and minimizer- regreg2", _regreg2, obj1(_regreg2[1]), obj2(_regreg2[1]))
        print("grad- scipy", grad(_scipy[1]))
        print("grad- regreg", grad(_regreg[1]))
        return obj1(_scipy[1]), _scipy[0], _regreg[0], _regreg2[0]

#test_dual_minimizations()
def test_one_sparse_dual_minimizations():

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
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))

    if nactive == 1:
        snr_seq = np.linspace(-10, 10, num=51)
        snr_seq = np.hstack([snr_seq[25:], snr_seq[:25][::-1]])
        lagrange = lam * np.ones(p)
        result = []
        for i in range(snr_seq.shape[0]):
            parameter = snr_seq[i]
            mean = X_1[:, active].dot(parameter)

            dual_scipy = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                         noise_variance, tau, epsilon)

            dual_regreg = selection_probability_dual_objective(X_1,
                                                               dual_feasible,
                                                               active,
                                                               active_signs,
                                                               lagrange,
                                                               mean,
                                                               noise_variance,
                                                               randomization.isotropic_gaussian((p,), tau),
                                                               epsilon)

            toc = time.time()
            dual_scipy_val = dual_scipy.minimize_dual()
            tic = time.time()
            print('scipy time', tic-toc)

            _scipy = [dual_scipy_val[0], dual_scipy_val[1]]

            toc = time.time()
            _regreg = dual_regreg.minimize(min_its=20, max_its=200, tol=1.e-12)[::-1]
            tic = time.time()
            print('regreg time', tic-toc)

            toc = time.time()
            _regreg2 = dual_regreg.minimize2()[::-1]
            tic = time.time()
            print('regreg2 time', tic-toc)

            obj1 = dual_scipy.dual_objective
            obj2 = lambda x : dual_regreg.smooth_objective(x, 'func')
            obj3 = lambda x : dual_regreg.objective(x)

            #check_two_approaches(_scipy[1], sel_prob_scipy, sel_prob_grad_descent)
            #check_two_approaches(_regreg[1], sel_prob_scipy, sel_prob_grad_descent)

            result.append([obj1(_scipy[1]), obj2(_scipy[1]), obj3(_scipy[1]),
                           obj1(_regreg[1]), obj2(_regreg[1]), obj3(_regreg[1]),
                           obj1(_regreg2[1]), obj2(_regreg2[1]), obj3(_regreg2[1])])

            print('scipy gradient', dual_regreg.smooth_objective(_scipy[1], 'grad'))
            print('regreg gradient', dual_regreg.smooth_objective(_regreg[1], 'grad'))

        return np.array(result)

def primal_dual_minimizations():

    #fixing n, p, true sparsity and signal strength
    n = 100
    p = 10
    s = 3
    snr = 5

    #sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    #getting randomized Lasso solution
    sel = selection(X_1,y, random_Z)
    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, active.sum())
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1 #randomization_variance
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(betaE)
    primal_feasible = np.fabs(betaE)
    #print('loc1')

    if nactive > 1:
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
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

        dual_regreg = selection_probability_dual_objective(X_1,
                                                           dual_feasible,
                                                           active,
                                                           active_signs,
                                                           lagrange,
                                                           mean,
                                                           noise_variance,
                                                           randomization.isotropic_gaussian((p,), tau),
                                                           epsilon)

        toc = time.time()
        primal_val = primal_regreg.minimize2(nstep=50, tol=1.e-8)[::-1]
        #primal_val = primal_regreg.minimize(max_its=1000, min_its=500, tol=1.e-12)[::-1]
        tic = time.time()
        print('primal time', tic-toc)

        primal_sol = primal_val[1]

        toc = time.time()
        dual_val = dual_regreg.minimize2(nstep=50, tol=1.e-8)[::-1]
        #dual_val = dual_regreg.minimize(max_its=1000, min_its=500, tol=1.e-12)[::-1]
        tic = time.time()
        print('dual time', tic-toc)

        dual_sol = mean - (dual_regreg.X_permute.dot(np.linalg.inv(dual_regreg.B_p.T))).dot(dual_val[1])

        print("value and minimizer- primal", primal_val)
        print("value and minimizer- dual", dual_val)
        print("primal and dual optimizer", primal_sol, dual_sol)

primal_dual_minimizations()