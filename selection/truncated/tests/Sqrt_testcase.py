"""
Sqrt_testcase.py
Date: 2014-10-17
Author: Xiaoying Tian
"""

from __future__ import division
from truncated_bis import (quadratic_inequality_solver, intersection, sqrt_inequality_solver)
import unittest
import numpy as np

class Sqrt_testcase(unittest.TestCase):

    def test_quadratic_solver(self):
        self.assertEqual(quadratic_inequality_solver(7,0,-28),[[-2.0,2.0]])
        self.assertEqual(quadratic_inequality_solver(1,-1,-5),[[-1.7912878474779199, 2.7912878474779199]])
        self.assertEqual(quadratic_inequality_solver(1,-1,5), [[]])
        self.assertEqual(quadratic_inequality_solver(-1,-1,-5), [[float("-inf"), float("inf")]])
        self.assertEqual(quadratic_inequality_solver(-1,6,-5), [[float("-inf"), 1.0], [5.0, float("inf")]])
        self.assertEqual(quadratic_inequality_solver(0,6,-5),[[float("-inf"), 0.8333333333333334]])
        self.assertEqual(quadratic_inequality_solver(0,6,5),[[float("-inf"), -0.8333333333333334]])
        self.assertRaises(ValueError, quadratic_inequality_solver, 0,0,5)
        self.assertEqual(quadratic_inequality_solver(1,3,2,"greater than"), [[float("-inf"), -2.], [-1., float("inf")]])

    def test_intersection(self):
        self.assertEqual(intersection([], []), [])
        self.assertEqual(intersection([], [1,2]), [])
        self.assertEqual(intersection([2,3], []), [])
        self.assertEqual(intersection([2,3], [1,2]), [])
        self.assertEqual(intersection([3,4], [1,2]), [])
        self.assertEqual(intersection([-1,4], [1,2]), [1,2])
        self.assertEqual(intersection([1,4], [-1,2]), [1,2])
        self.assertEqual(intersection([1,4], [-1,12]), [1,4])

    def test_sqrt_solver(self):
        a, b, c = np.random.random_integers(-50, 50, 3)
        n = 100
        intervals = sqrt_inequality_solver(a, b, c, n)
        print a, b, c, intervals
        for x in np.linspace(-20, 20):
            hold = (func(x, a, b, c, n) <= 0)
            in_interval = any([contains(x, I) for I in intervals])
            self.assertEqual(hold, in_interval)
        

def contains(x, I):
    if I:
        return (x >= I[0] and x <= I[1])
    else:
        return False


def func(x, a, b, c, n):
    return a*x + b * np.sqrt(n + x**2) - c 


if __name__ == "__main__":
    unittest.main()
