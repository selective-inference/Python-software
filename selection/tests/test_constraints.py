import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import chi
import nose.tools as nt

import selection.affine as AC
reload(AC)

def test_conditional():

    p = 200
    k1, k2 = 5, 3
    b = np.random.standard_normal((k1,))
    A = np.random.standard_normal((k1,p))
    con = AC.constraints(A,b)
    w = np.random.standard_normal(p)
    con.mean = w
    C = np.random.standard_normal((k2,p))
    d = np.random.standard_normal(k2)
    new_con = con.conditional(C, d)

    while True:
        W = np.random.standard_normal(p)
        W -= np.dot(np.linalg.pinv(C), np.dot(C, W) - d)  
        if new_con(W) and con(W):
            break

    Z = AC.sample_from_constraints(new_con, W, ndraw=5000)

    tol = 0
    
    nt.assert_true(np.linalg.norm(np.dot(Z, C.T) - d[None,:]) < 1.e-7)
    Zn = Z - new_con.translate[None,:]
    V = (np.dot(Zn, new_con.linear_part.T) - new_con.offset[None,:]).max(1)
    V2 = (np.dot(Z, con.linear_part.T) - con.offset[None,:]).max(1)
    print ('failing:', 
           (V>tol).sum(), 
           (V2>tol).sum(), 
           np.linalg.norm(np.dot(C, W) - d))
    nt.assert_true(np.sum(V > tol) < 0.001*V.shape[0])


def test_conditional_simple():

    A = np.ones((1,2))
    b = np.array([1])
    con = AC.constraints(A,b)

    C = np.array([[0,1]])
    d = np.array([2])

    new_con = con.conditional(C,d)

    while True:
        W = np.random.standard_normal(2)
        W -= np.dot(np.linalg.pinv(C), np.dot(C, W) - d)  
        if con(W):
            break
    Z1 = AC.simulate_from_constraints(new_con, W, ndraw=10000)

    counter = 0
    new_sample = []
    while True:
        W = np.random.standard_normal() + 2 # conditional distribution
        if W < 1:
            new_sample.append(W)
            counter += 1

        if counter >= 10000:
            break

    return Z1, np.array(new_sample)

def test_stack():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con1 = AC.constraints((A,b), (E,f))

    A, b = np.random.standard_normal((5,30)), np.random.standard_normal(5)
    E, f = np.random.standard_normal((3,30)), np.random.standard_normal(3)

    con2 = AC.constraints((A,b), (E,f))

    return AC.stack(con1, con2)

def test_simulate():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = AC.constraints((A,b), (E,f))
    return con, AC.simulate_from_constraints(con)

def test_simulate_nonwhitened():
    n, p = 50, 200

    X = np.random.standard_normal((n,p))
    cov = np.dot(X.T, X)

    W = np.random.standard_normal((3,p))
    con = AC.constraints(W, np.ones(3), covariance=cov)

    while True:
        z = np.random.standard_normal(p)
        if np.dot(W, z).max() <= 1:
            break

    Z = AC.simulate_from_constraints(con, z)
    nt.assert_true((np.dot(Z, W.T) - 1).max() < 0)

def test_pivots():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = AC.constraints((A,b), (E,f))
    Z = AC.simulate_from_constraints(con)
    u = np.zeros(con.dim)
    u[4] = 1
    return con.pivots(u, Z)

def test_pivots2():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    E, f = np.random.standard_normal((2,30)), np.random.standard_normal(2)

    con = AC.constraints((A,b), (E,f))
    nsim = 10000
    u = np.zeros(con.dim)
    u[4] = 1

    P = []
    for i in range(nsim):
        Z = AC.simulate_from_constraints(con)
        P.append(con.pivots(u, Z)[-1])
    P = np.array(P)
    P = P[P > 0]
    P = P[P < 1]
    return P

def test_chisq_central():

    A, b = np.random.standard_normal((4,6)), np.zeros(4)
    con = AC.constraints((A,b), None)

    nsim = 3000
    P = []
    for i in range(nsim):
        Z = AC.simulate_from_constraints(con)
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
    con = AC.constraints((A,b), None)

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
        Z = AC.simulate_from_constraints(con,mu=mu)
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
