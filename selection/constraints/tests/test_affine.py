from __future__ import absolute_import, print_function

import nose
import numpy as np
from scipy.stats import chi
import nose.tools as nt

import regreg.api as rr

from selection.tests.flags import SET_SEED
import selection.constraints.affine as AC
from selection.tests.decorators import set_seed_iftrue

@set_seed_iftrue(SET_SEED)
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

    V = (np.dot(Z, new_con.linear_part.T) - new_con.offset[None,:]).max(1)
    V2 = (np.dot(Z, con.linear_part.T) - con.offset[None,:]).max(1)
    print ('failing:', 
           (V>tol).sum(), 
           (V2>tol).sum(), 
           np.linalg.norm(np.dot(C, W) - d))
    nt.assert_true(np.sum(V > tol) < 0.001*V.shape[0])

@set_seed_iftrue(SET_SEED)
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
    nt.assert_true(test < 5)

def test_stack():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)

    con1 = AC.constraints(A,b)

    A, b = np.random.standard_normal((5,30)), np.random.standard_normal(5)
    E, f = np.random.standard_normal((3,30)), np.random.standard_normal(3)

    con2 = AC.constraints(A,b)

    return AC.stack(con1, con2)

def test_regreg_transform():

    A, b = np.random.standard_normal((4,30)), np.random.standard_normal(4)
    A = rr.astransform(A)
    con = AC.constraints(A,b, covariance=np.identity(30))

    while True:
        Z = np.random.standard_normal(30) # conditional distribution
        if con.value(Z) < 0:
            break

    C = np.random.standard_normal((2,30))
    conditional = con.conditional(C, C.dot(Z))
    W = np.random.standard_normal(30)

    print(conditional.pivot(W, Z))
    print(con.pivot(W, Z))

@set_seed_iftrue(SET_SEED)
def test_simulate_nonwhitened():
    n, p = 50, 200

    X = np.random.standard_normal((n,p))
    cov = np.dot(X.T, X)

    W = np.random.standard_normal((3,p))
    con = AC.constraints(W, 3 * np.ones(3), covariance=cov)

    while True:
        z = np.random.standard_normal(p)
        if np.dot(W, z).max() <= 3:
            break

    Z = AC.sample_from_constraints(con, z, burnin=100, ndraw=100)
   
    nt.assert_true((np.dot(Z, W.T) - 3).max() < 1.e-5)

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

@set_seed_iftrue(SET_SEED)
def test_sampling():
    """
    See that means and covariances are approximately correct
    """
    C = AC.constraints(np.identity(3), np.inf*np.ones(3))
    C.mean = np.array([3,4,5.2])
    W = np.random.standard_normal((5,3))
    S = np.dot(W.T, W) / 30.
    C.covariance = S
    V = AC.sample_from_constraints(C, np.zeros(3), ndraw=500000)

    nt.assert_true(np.linalg.norm(V.mean(0)-C.mean) < 0.01)
    nt.assert_true(np.linalg.norm(np.einsum('ij,ik->ijk', V, V).mean(0) - 
                                  np.outer(V.mean(0), V.mean(0)) - S) < 0.01)

@set_seed_iftrue(SET_SEED)
@np.testing.decorators.skipif(True, msg="optimal tilt undefined -- need to implement softmax version")
def test_optimal_tilt():

    A = np.vstack(-np.identity(4))
    b = -np.array([1,2,3,4.])
    con = AC.constraints(A, b, covariance=2 * np.identity(4),
                     mean=np.array([3,-4.3,-2.2,1.2]))
    eta = np.array([1,0,0,0.])

    tilt = optimal_tilt(con, eta)
    print(tilt.smooth_objective(np.zeros(5), mode='both'))
    opt_tilt = tilt.fit(max_its=20)
    print(con.mean + opt_tilt)

    A = np.vstack([-np.identity(4),
                    np.identity(4)])
    b = np.array([-1,-2,-3,-4.,3,4,5,11])
    con = AC.constraints(A, b, covariance=2 * np.identity(4),
                     mean=np.array([3,-4.3,12.2,20.2]))
    eta = np.array([1,0,0,0.])

    tilt = optimal_tilt(con, eta)
    print(tilt.smooth_objective(np.zeros(5), mode='both'))
    opt_tilt = tilt.fit(max_its=20)
    print(con.mean + opt_tilt)

