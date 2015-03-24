import numpy as np
from selection.screening import positive_screen
import nose.tools as nt

def test_class(threshold=1):
    
    Z = np.random.standard_normal(10)
    C = np.eye(10)
    M = positive_screen(Z, C, 1)
    M.constraints

    np.testing.assert_allclose(np.dot(M.constraints.inequality, M.Z) + M.constraints.inequality_offset, M.Z[M.selected] - M.threshold)
    np.testing.assert_array_less(np.zeros(M.constraints.inequality.shape[0]), np.dot(M.constraints.inequality, M.Z) + M.constraints.inequality_offset)

    M.intervals
    return M

