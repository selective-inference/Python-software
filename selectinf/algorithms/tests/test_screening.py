import numpy as np
from ..screening import topK
import nose.tools as nt

def test_class(threshold=1):
    
    Z = np.random.standard_normal(10)
    C = np.eye(10)
    M = topK(C, Z, 1, 1)
    M.constraints

    M.intervals
    return M

