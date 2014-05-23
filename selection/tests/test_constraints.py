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
    con = AC.constraints(A,b) #X1+X2<= 1

    C = np.array([[0,1]])
    d = np.array([2])   #X2=2

    new_con = con.conditional(C,d)
    while True:
        W = np.random.standard_normal(2)
        W -= np.dot(np.linalg.pinv(C), np.dot(C, W) - d)  
        if con(W):
            break
    Z1 = AC.sample_from_constraints(new_con, W, ndraw=10000)

    counter = 0
    new_sample = []
    while True:
        W = np.random.standard_normal() # conditional distribution
        if W < -1:
            new_sample.append(W)
            counter += 1

        if counter >= 10000:
            break

    a1 = Z1[:,0]
    a2 = np.array(new_sample)
    test = np.fabs((a1.mean() - a2.mean()) / (np.std(a1) * np.sqrt(2)) * np.sqrt(10000))
    print 'test'
    nt.assert_true(test < 5)

def test_stack():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)

    con1 = AC.constraints(A,b)

    A, b = np.random.standard_normal((5,30)), np.random.standard_normal(5)
    E, f = np.random.standard_normal((3,30)), np.random.standard_normal(3)

    con2 = AC.constraints(A,b)

    return AC.stack(con1, con2)


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

    Z = AC.sample_from_constraints(con, z)
    nt.assert_true((np.dot(Z, W.T) - 1).max() < 0)

def test_pivots_intervals():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)

    con = AC.constraints(A,b)
    while True:
        w = np.random.standard_normal(30)
        if con(w):
            break

    Z = AC.sample_from_constraints(con, w)[-1]
    u = np.zeros(con.dim)
    u[4] = 1

    # call pivot
    con.pivot(u, Z)
    con.pivot(u, Z, alternative='less')
    con.pivot(u, Z, alternative='greater')

    con.interval(u, Z, UMAU=True)
    con.interval(u, Z, UMAU=False)

