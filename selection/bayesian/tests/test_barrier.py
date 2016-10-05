from __future__ import print_function
from scipy.optimize import minimize
import numpy as np
import regreg.api as rr

import selection.bayesian.sel_probability2;
from imp import reload
reload(selection.bayesian.sel_probability2)
from selection.bayesian.sel_probability2 import cube_subproblem, cube_gradient, cube_barrier, selection_probability_objective
from selection.randomized.api import randomization
from selection.bayesian.barrier import barrier_conjugate

def test_barrier_conjugate():

    p = 10
    cube_bool = np.zeros(p, np.bool)
    cube_bool[:4] = 1
    _barrier_star = barrier_conjugate(cube_bool,
                                      np.arange(4) + 1)

    arg = np.zeros(p)
    arg[cube_bool] = np.random.standard_normal(4)
    arg[~cube_bool] = - np.fabs(np.random.standard_normal(6))
    print(_barrier_star.smooth_objective(arg, mode='both'))
    
    B1 = np.random.standard_normal((4,10))

    # linear composition
    composition1 = rr.affine_smooth(_barrier_star, B1.T)
    print(composition1.shape)

    X = np.random.standard_normal((10, 4))
    Y = np.random.standard_normal(10)
    A = rr.affine_transform(X, Y)

    composition2 = rr.affine_smooth(_barrier_star, A)
    print(composition2.shape)

    sum_fn = rr.smooth_sum([composition1, composition2])
    print(sum_fn.shape)

def test_cube_subproblem(k=100, do_scipy=True, verbose=False):

    k = 100
    randomization_variance = 0.5
    lagrange = 2
    conjugate_argument = 0.5 * np.random.standard_normal(k)
    conjugate_argument[5:] *= 2

    randomizer = randomization.isotropic_gaussian((k,), 2.)

    conj_value, conj_grad = randomizer.CGF_conjugate

    soln, val = cube_subproblem(conjugate_argument,
                                randomizer.CGF_conjugate,
                                lagrange, nstep=10,
                                lipschitz=randomizer.lipschitz)

    if do_scipy:
        objective = lambda x: cube_barrier(x, lagrange) + conj_value(conjugate_argument + x)
        scipy_soln = minimize(objective, x0=np.zeros(k)).x

        if verbose:
            print('scipy soln', scipy_soln)
            print('newton soln', soln)

            print('scipy val', objective(scipy_soln))
            print('newton val', objective(soln))

            print('scipy grad', cube_gradient(scipy_soln, lagrange) + 
                  conj_grad(conjugate_argument + scipy_soln))
            print('newton grad', cube_gradient(soln, lagrange) + 
                  conj_grad(conjugate_argument + soln))
        if (objective(soln) > objective(scipy_soln) + 0.01 * np.fabs(objective(soln))):
            raise ValueError('scipy won!')

def test_cube_laplace(k=100, do_scipy=True, verbose=False):

    k = 100
    randomization_variance = 0.5
    lagrange = 2
    conjugate_argument = 0.5 * np.random.standard_normal(k)
    conjugate_argument[5:] *= 2

    randomizer = randomization.laplace((k,), 2.)

    conj_value, conj_grad = randomizer.CGF_conjugate

    soln, val = cube_subproblem(conjugate_argument,
                                randomizer.CGF_conjugate,
                                lagrange, nstep=10,
                                lipschitz=randomizer.lipschitz)

    if do_scipy:
        objective = lambda x: cube_barrier(x, lagrange) + conj_value(conjugate_argument + x)
        scipy_soln = minimize(objective, x0=np.zeros(k)).x

        if verbose:
            print('scipy soln', scipy_soln)
            print('newton soln', soln)

            print('scipy val', objective(scipy_soln))
            print('newton val', objective(soln))

            print('scipy grad', cube_gradient(scipy_soln, lagrange) + 
                  conj_grad(conjugate_argument + scipy_soln))
            print('newton grad', cube_gradient(soln, lagrange) + 
                  conj_grad(conjugate_argument + soln))
        if (objective(soln) > objective(scipy_soln) + 0.01 * np.fabs(objective(soln))):
            raise ValueError('scipy won!')

def test_selection_probability():

    n, p, s = 50, 100, 5

    X = np.random.standard_normal((n, p))

    scalings = np.ones(s)

    active = np.zeros(p, np.bool)
    active[:s] = 1
    np.random.shuffle(active)

    active_signs = ((-1)**np.arange(p))
    np.random.shuffle(active_signs)
    active_signs = active_signs[:s]

    lagrange = np.ones(p) + np.linspace(-0.1,0.1,p)

    parameter = np.ones(s) 

    noise_variance = 1.5

    sel_prob = selection_probability_objective(X,
                                               scalings,
                                               active,
                                               active_signs,
                                               lagrange,
                                               X[:,active].dot(parameter),
                                               noise_variance,
                                               randomization.laplace((p,), 2.))

    sel_prob.minimize()

def test_selection_probability_gaussian():

    n, p, s = 50, 100, 5

    X = np.random.standard_normal((n, p))

    scalings = np.ones(s)

    active = np.zeros(p, np.bool)
    active[:s] = 1
    np.random.shuffle(active)

    active_signs = ((-1)**np.arange(p))
    np.random.shuffle(active_signs)
    active_signs = active_signs[:s]

    lagrange = np.ones(p) + np.linspace(-0.1,0.1,p)

    parameter = np.ones(s) 

    noise_variance = 1.5

    sel_prob = selection_probability_objective(X,
                                               scalings,
                                               active,
                                               active_signs,
                                               lagrange,
                                               X[:,active].dot(parameter),
                                               noise_variance,
                                               randomization.isotropic_gaussian((p,), 2.))

    sel_prob.minimize()


