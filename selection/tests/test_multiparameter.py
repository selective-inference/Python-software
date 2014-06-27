import numpy as np
import selection.discrete_multiparameter
reload(selection.discrete_multiparameter)
from selection.discrete_multiparameter import multiparameter_family

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

