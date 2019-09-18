import numpy as np
from ..discrete_multiparameter import multiparameter_family

def test_multiparameter():

    X = [[3,4],[4,5],[5,8.]]
    w = [0.3, 0.5, 0.4]
    theta = [0.1,0.3]

    family = multiparameter_family(X, w)
    mu1 = family.mean(theta)

    X_arr = np.array(X)
    exponent = np.dot(X_arr, theta)

    w_arr = np.array(w) * np.exp(exponent)
    w_arr /= w_arr.sum()

    mu2 = (X_arr * w_arr[:,None]).sum(0)

    np.testing.assert_allclose(mu1, mu2)

    info1 = family.information(theta)[1]

    T = np.zeros((3,2,2))
    for i in range(2):
        for j in range(2):
            T[:,i,j] = X_arr[:,i] * X_arr[:,j]
        
    second_moment = (T * w_arr[:,None,None]).sum(0)
    info2 = second_moment - np.outer(mu1, mu1)
    
    np.testing.assert_allclose(info1, info2)

    mu3 = np.array([family.E(theta, lambda x: x[:,i]) for i in range(2)])
    np.testing.assert_allclose(mu1, mu3)

    cov01 = np.array(family.Cov(theta, lambda x: x[:,0], lambda x: x[:,1]))
    np.testing.assert_allclose(cov01, info1[0,1])

    var0 = np.array(family.Var(theta, lambda x: x[:,0]))
    np.testing.assert_allclose(var0, info1[0,0])

    observed = np.array([4.2,6.3])
    theta_hat = family.MLE(observed, tol=1.e-12, max_iters=50)

    np.testing.assert_allclose(observed, family.mean(theta_hat))
