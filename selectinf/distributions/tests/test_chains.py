import numpy as np

from ..chain import parallel_test, serial_test
from ...constraints.affine import constraints, gaussian_hit_and_run

def test_gaussian_chain():

    n = 30

    A = np.eye(n)[:3]
    b = np.ones(A.shape[0])

    con = constraints(A, b)
    state = np.random.standard_normal(n)
    state[:3] = 0

    gaussian_chain = gaussian_hit_and_run(con, state, nstep=100)

    counter = 0
    for step in gaussian_chain:
        counter += 1
        
        if counter >= 100:
            break

    test_statistic = lambda z: np.sum(z)

    parallel = parallel_test(gaussian_chain, 
                             gaussian_chain.state,
                             test_statistic,
                             ndraw=20)

    serial = serial_test(gaussian_chain, 
                         gaussian_chain.state,
                         test_statistic,
                         ndraw=20)

    return parallel, serial
