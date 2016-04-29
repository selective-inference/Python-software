import numpy as np
from selection.constraints.optimal_tilt import (_solve_softmax_problem,
                                                softmax,
                                                softmax_conjugate)
from selection.constraints.affine import constraints

def test_softmax():

    A = np.array([[-1.,0.]])
    b = np.array([-1.])

    con = constraints(A, b)
    observed = np.array([1.4,2.3])

    simple_estimator = con.estimate_mean(observed)
    f, coefs = _solve_softmax_problem(simple_estimator, con, observed)

    np.testing.assert_allclose(coefs, observed)

    loss = softmax_conjugate(con,
                             observed)
    np.testing.assert_allclose(loss.smooth_objective(simple_estimator, 'grad'), observed)

    loss.smooth_objective(coefs, 'both')
