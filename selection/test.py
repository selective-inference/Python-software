import numpy as np
import nose.tools as nt
import constraints as C; reload(C)
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import chi, ncx2, chi2

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri
ro.numpy2ri.activate()

def test_apply_equality():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con1 = C.constraints((A,b), (E, f))
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

    con1 = C.constraints((A,b), (E,f))

    A, b = np.random.standard_normal((5,30)), np.random.standard_normal(5)
    E, f = np.random.standard_normal((3,30)), np.random.standard_normal(3)

    con2 = C.constraints((A,b), (E,f))

    return C.stack(con1, con2)

def test_simulate():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = C.constraints((A,b), (E,f))
    return con, C.simulate_from_constraints(con)

def test_pivots():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = C.constraints((A,b), (E,f))
    Z = C.simulate_from_constraints(con)
    u = np.zeros(con.dim)
    u[4] = 1
    return con.pivots(u, Z)

def test_pivots2():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = C.constraints((A,b), (E,f))
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

def test_chisq_central():

    A, b = np.random.standard_normal((4,6)), np.zeros(4)
    con = C.constraints((A,b), None)

    nsim = 3000
    P = []
    for i in range(nsim):
        Z = C.simulate_from_constraints(con)
        u = 0 * Z
        u[:3] = Z[:3] / np.linalg.norm(Z[:3])
        L, V, U = con.pivots(u, Z)[:3]
        ncp = 1.e-3
        P.append((chi.cdf(U,3) - chi.cdf(V,3)) 
                 / (chi.cdf(U,3) - chi.cdf(L,3)))

    ecdf = sm.distributions.ECDF(P)

    plt.clf()
    x = np.linspace(0,1,101)
    plt.plot(x, ecdf(x), c='red')
    plt.plot([0,1],[0,1], c='blue', linewidth=2)

def test_chisq_noncentral():

    mu = np.arange(6)
    ncp = np.linalg.norm(mu[:3])**2

    A, b = np.random.standard_normal((4,6)), np.zeros(4)
    con = C.constraints((A,b), None)

    ro.r('fncp=%f' % ncp)
    ro.r('f = function(x) {pchisq(x,3,ncp=fncp)}')
    def F(x):
        if x != np.inf:
            return np.array(ro.r('f(%f)' % x))
        else:
            return np.array([1.])

    nsim = 2000
    P = []
    for i in range(nsim):
        Z = C.simulate_from_constraints(con,mu=mu)
        print i
        u = 0 * Z
        u[:3] = Z[:3] / np.linalg.norm(Z[:3])
        L, V, U = con.pivots(u, Z)[:3]
        if L > 0:
            Ln = L**2
            Un = U**2
            Vn = V**2
        else:
            Ln = 0
            Un = U**2
            Vn = V**2

        if U < 0:
            stop
        P.append(np.array((F(Un) - F(Vn)) / (F(Un) - F(Ln))))

    P = np.array(P).reshape(-1)
    P = P[P > 0]
    P = P[P < 1]

    ecdf = sm.distributions.ECDF(P)

    plt.clf()
    x = np.linspace(0,1,101)
    plt.plot(x, ecdf(x), c='red')
    plt.plot([0,1],[0,1], c='blue', linewidth=2)
