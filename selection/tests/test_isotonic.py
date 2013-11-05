from ..isotonic import isotonic
import numpy as np

def test_isotonic():
    y = np.random.standard_normal(50)
    I = isotonic(y)
    print I.first_jump
    print I.largest_jump
    print I.combine_jumps(2)
