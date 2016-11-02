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

###primal problem
#deciding how close regreg is to scipy
def primal_speed():
    n = 100
    p = 20
    s = 10
    snr = 5

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, true_beta)
    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    if nactive > 1:
        #parameter = true_beta[active]
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
        print('scipy time', tic - toc)

        sel_prob_regreg = selection_probability_objective(X_1,
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
        _regreg = sel_prob_regreg.minimize(max_its=100, tol=1.e-12)[::-1]
        tic = time.time()
        print('regreg time', tic - toc)

        obj1 = sel_prob_scipy.objective
        obj2 = lambda x: sel_prob_regreg.smooth_objective(x, 'func')

        toc = time.time()
        _regreg2 = sel_prob_regreg.minimize2(nstep=20)[::-1]
        tic = time.time()
        print('regreg2', tic - toc)

        test = np.ones(n+nactive)
        test[n:] = np.fabs(np.random.standard_normal(nactive))

        #print("vals and minimizers", _scipy, _regreg, _regreg2)
        print("value and minimizer- scipy",  obj1(_scipy[1]), obj2(_scipy[1]))
        print("value and minimizer- regreg1", obj1(_regreg[1]), obj2(_regreg[1]))
        print("value and minimizer- regreg2", obj1(_regreg2[1]), obj2(_regreg2[1]))

        print('check objectives', obj1(test), obj2(test))
        np.testing.assert_allclose(sel_prob_scipy.objective(test),
                                   sel_prob_regreg.smooth_objective(test, 'func'), rtol=1.e-5)

        return _scipy[0], _regreg[0]

#primal_speed()

#deciding amongst minimize2 and minimize
def regreg_test():
    n = 600
    p = 90
    s = 10
    snr = 5

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, true_beta)
    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    if nactive > 1:
        # parameter = true_beta[active]
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)

        sel_prob_regreg = selection_probability_objective(X_1,
                                                          np.fabs(betaE),
                                                          active,
                                                          active_signs,
                                                          lagrange,
                                                          mean,
                                                          noise_variance,
                                                          randomization.isotropic_gaussian((p,), 1.),
                                                          epsilon,
                                                          nstep=10)

        obj = lambda x: sel_prob_regreg.smooth_objective(x, 'func')

        toc = time.time()
        _regreg = sel_prob_regreg.minimize(max_its=100, tol=1.e-10)[::-1]
        tic = time.time()
        print('regreg', tic - toc)

        toc = time.time()
        _regreg2 = sel_prob_regreg.minimize2(nstep=20)[::-1]
        tic = time.time()
        print('regreg2', tic - toc)

        print("vals", _regreg2[0], _regreg[0])
        print("vals", obj(_regreg2[1]), obj(_regreg[1]))

        return  _regreg2[0], _regreg[0]

#regreg_test()

#checking if number of iterations in minimize2 make a difference (there are few cases when regreg2 doesnt quite converge with 40 steps)
def regreg_iterations_test():
    n = 600
    p = 90
    s = 10
    snr = 5

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, true_beta)
    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    if nactive > 1:
        # parameter = true_beta[active]
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)

        sel_prob_regreg = selection_probability_objective(X_1,
                                                          np.fabs(betaE),
                                                          active,
                                                          active_signs,
                                                          lagrange,
                                                          mean,
                                                          noise_variance,
                                                          randomization.isotropic_gaussian((p,), 1.),
                                                          epsilon,
                                                          nstep=10)

        obj = lambda x: sel_prob_regreg.smooth_objective(x, 'func')

        toc = time.time()
        _regreg1 = sel_prob_regreg.minimize2(nstep=80)[::-1]
        tic = time.time()
        print('regreg1', tic - toc)

        toc = time.time()
        _regreg2 = sel_prob_regreg.minimize2(nstep=40)[::-1]
        tic = time.time()
        print('regreg2', tic - toc)

        print("vals", _regreg2[0], _regreg1[0])
        print("vals", obj(_regreg2[1]), obj(_regreg1[1]))

        return  _regreg2[0], _regreg1[0]

#regreg_iterations_test()

###dual problem
#checking how close dual is to primal

#low dim regime same as HIV data set
def dual_primal_speed_low():
    n = 600
    p = 90
    s = 10
    snr = 5

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, betaE.shape[0])
    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
    primal_feasible = np.fabs(betaE)

    if nactive > 1:
        #parameter = true_beta[active]
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
        _regreg_dual = dual_regreg.minimize(max_its=1000, min_its=50, tol=1.e-10)[::-1]
        tic = time.time()
        print('dual time', tic - toc)

        toc = time.time()
        _regreg2_dual = dual_regreg.minimize2(nstep=50)[::-1]
        tic = time.time()
        print('dual time', tic - toc)

        toc = time.time()
        _regreg = primal_regreg.minimize2(nstep=50)[::-1]
        tic = time.time()
        print('primal time', tic - toc)

        print(_regreg_dual[0], _regreg2_dual[0], _regreg[0])
    return _regreg[0]


#dual_primal_speed_low()

#high dim regime
def dual_primal_speed_high():
    n = 100
    p = 50
    s = 40
    snr = 5

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, betaE.shape[0])
    noise_variance = 1
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
    primal_feasible = np.fabs(betaE)

    if nactive > 1:
        #parameter = true_beta[active]
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
        _regreg_dual = dual_regreg.minimize(max_its=1000, min_its=50, tol=1.e-10)[::-1]
        tic = time.time()
        print('dual time', tic - toc)

        toc = time.time()
        _regreg2_dual = dual_regreg.minimize2(nstep=50)[::-1]
        tic = time.time()
        print('dual time 2', tic - toc)

        toc = time.time()
        _regreg = primal_regreg.minimize2(nstep=50)[::-1]
        tic = time.time()
        print('primal time', tic - toc)

        print(_regreg_dual[0], _regreg2_dual[0], _regreg[0])
    return _regreg[0]


dual_primal_speed_high()

