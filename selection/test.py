import numpy as np
import nose.tools as nt
import constraints as C; reload(C)

def test_apply_equality():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con1 = C.constraint((A,b), (E, f))
    con2 = con1.impose_equality()
    con3 = con2.impose_equality()

    np.testing.assert_allclose(con1.equality, 
                               con2.equality)
    np.testing.assert_allclose(con1.equality_offset, 
                               con2.equality_offset)

    np.testing.assert_allclose(con1.equality, 
                               con3.equality)
    np.testing.assert_allclose(con1.equality_offset, 
                               con3.equality_offset)

    np.testing.assert_allclose(con2.inequality, 
                               con3.inequality)
    np.testing.assert_allclose(con2.inequality_offset, 
                               con3.inequality_offset)

def test_stack():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con1 = C.constraint((A,b), (E,f))

    A, b = np.random.standard_normal((5,30)), np.random.standard_normal(5)
    E, f = np.random.standard_normal((3,30)), np.random.standard_normal(3)

    con2 = C.constraint((A,b), (E,f))

    return C.stack(con1, con2)

def test_simulate():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = C.constraint((A,b), (E,f))
    return con, C.simulate_from_constraints(con)

def test_pivots():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = C.constraint((A,b), (E,f))
    Z = C.simulate_from_constraints(con)
    u = np.zeros(con.dim)
    u[4] = 1
    return con.pivots(u, Z)

def test_pivots2():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = C.constraint((A,b), (E,f))
    nsim = 10000
    u = np.zeros(con.dim)
    u[4] = 1

    P = []
    for i in range(nsim):
        Z = C.simulate_from_constraints(con)
        P.append(con.pivots(u, Z)[-1])
    P = np.array(P)
    P = P[P > 0]
    P = P[P < 1]
    return P
