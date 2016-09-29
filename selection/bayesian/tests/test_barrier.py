from __future__ import print_function
import numpy as np

import selection.bayesian.sel_probability2; 
from imp import reload
reload(selection.bayesian.sel_probability2)
from selection.bayesian.sel_probability2 import subgradient_subproblem, cube_gradient, cube_barrier

def test_subgradient_subproblem():

    k = 10
    randomization_variance = 0.5
    lagrange = 2
    conjugate_argument = 5 * np.random.standard_normal(k)

    soln = subgradient_subproblem(conjugate_argument,
                                  randomization_variance,
                                  lagrange)

    print(cube_gradient(soln, lagrange) - 
          conjugate_argument + 
          soln / randomization_variance)
