"""
test_quasi.py
Date: 2014-10-17
Author: Xiaoying Tian
"""

from __future__ import division, print_function
import nose.tools as nt
import numpy as np

from selection.constraints.quasi_affine import (quadratic_inequality_solver, intersection, sqrt_inequality_solver)
from selection.tests.flags import SET_SEED
from selection.tests.decorators import set_seed_iftrue

def test_quadratic_solver():
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(7,0.,-28),[[-2.0,2.0]]
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(1,-1,-5.),[[-1.7912878474779199, 2.7912878474779199]]
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(1,-1,5.), [[]]
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(-1,-1,-5.), [[float("-inf"), float("inf")]]
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(-1,6,-5.), [[float("-inf"), 1.0], [5.0, float("inf")]]
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(0.,6,-5.),[[float("-inf"), 0.8333333333333334]]
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(0.,6,5.),[[float("-inf"), -0.8333333333333334]]
    yield nt.assert_raises, ValueError, quadratic_inequality_solver, 0., 0., 5.
    yield np.testing.assert_almost_equal, quadratic_inequality_solver(1,3,2,"greater than"), [[float("-inf"), -2.], [-1., float("inf")]]

def test_intersection():
    yield np.testing.assert_almost_equal, intersection([], []), []
    yield np.testing.assert_almost_equal, intersection([], [1,2]), []
    yield np.testing.assert_almost_equal, intersection([2,3], []), []
    yield np.testing.assert_almost_equal, intersection([2,3], [1,2]), []
    yield np.testing.assert_almost_equal, intersection([3,4], [1,2]), []
    yield np.testing.assert_almost_equal, intersection([-1,4], [1,2]), [1,2]
    yield np.testing.assert_almost_equal, intersection([1,4], [-1,2]), [1,2]
    yield np.testing.assert_almost_equal, intersection([1,4], [-1,12]), [1,4]

@set_seed_iftrue(SET_SEED)
def test_sqrt_solver():
    a, b, c = np.random.random_integers(-50, 50, 3)
    n = 100
    intervals = sqrt_inequality_solver(a, b, c, n)
    print(a, b, c, intervals)
    for x in np.linspace(-20, 20):
        hold = (func(x, a, b, c, n) <= 0)
        in_interval = any([contains(x, I) for I in intervals])
        yield np.testing.assert_almost_equal, np.array(hold, np.float), np.array(in_interval, np.float)


def contains(x, I):
    if I:
        return (x >= I[0] and x <= I[1])
    else:
        return False


def func(x, a, b, c, n):
    return a*x + b * np.sqrt(n + x**2) - c 

